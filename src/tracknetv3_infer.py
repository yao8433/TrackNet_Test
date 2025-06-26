"""
TrackNetV3 inference module - implementing real inference pipeline
Based on TrackNetV3 PyTorch implementations from GitHub repositories
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Tuple, List
import torchvision.transforms as transforms

# Default model paths
MODEL_PATH = "models/tracknetv3/TrackNet_best.pt"
INPAINT_PATH = "models/tracknetv3/InpaintNet_best.pt"

# Model parameters based on actual TrackNetV3 checkpoint
IMG_HEIGHT = 288
IMG_WIDTH = 512
SEQUENCE_LENGTH = 8  # Based on param_dict: seq_len = 8
INPUT_CHANNELS = 27  # 9 frames × 3 RGB channels


class ConvBlock(nn.Module):
    """Convolutional block with conv + bn + relu"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DownBlock(nn.Module):
    """Down sampling block with two conv blocks + maxpool"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv_1 = ConvBlock(in_channels, out_channels)
        self.conv_2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return self.pool(x), x  # Return pooled and unpooled for skip connections


class UpBlock(nn.Module):
    """Up sampling block with transpose conv"""
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_1 = ConvBlock(in_channels, out_channels)  # in_channels because of skip connection
        self.conv_2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class TrackNetV3(nn.Module):
    """
    TrackNetV3 architecture matching the actual model checkpoint structure.
    """
    def __init__(self, input_channels=27, output_channels=1):
        super(TrackNetV3, self).__init__()
        
        # Encoder (down blocks)
        self.down_block_1 = DownBlock(input_channels, 64)
        self.down_block_2 = DownBlock(64, 128)
        self.down_block_3 = DownBlock(128, 256)
        self.down_block_4 = DownBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(512, 1024),
            ConvBlock(1024, 1024)
        )
        
        # Decoder (up blocks)
        self.up_block_1 = UpBlock(1024, 512)
        self.up_block_2 = UpBlock(512, 256)
        self.up_block_3 = UpBlock(256, 128)
        self.up_block_4 = UpBlock(128, 64)
        
        # Final prediction layer
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Encoder path
        x1, skip1 = self.down_block_1(x)
        x2, skip2 = self.down_block_2(x1)
        x3, skip3 = self.down_block_3(x2)
        x4, skip4 = self.down_block_4(x3)
        
        # Bottleneck
        x = self.bottleneck(x4)
        
        # Decoder path with skip connections
        x = self.up_block_1(x, skip4)
        x = self.up_block_2(x, skip3)
        x = self.up_block_3(x, skip2)
        x = self.up_block_4(x, skip1)
        
        # Final prediction
        x = self.final_conv(x)
        return self.sigmoid(x)


def load_tracknetv3_model(model_path=MODEL_PATH, device='cpu'):
    """
    Load TrackNetV3 model from the specified path.
    
    Args:
        model_path: Path to the model weights file
        device: Device to load model on ('cpu' or 'cuda')
        
    Returns:
        Loaded model object
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading TrackNetV3 model from: {model_path}")
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize model with correct input channels
        model = TrackNetV3(input_channels=INPUT_CHANNELS, output_channels=1)
        
        # Load the model state dict (it's stored under 'model' key)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print("✓ Loaded model from 'model' key in checkpoint")
        else:
            # Fallback to direct loading
            model.load_state_dict(checkpoint)
            print("✓ Loaded model as direct state dict")
        
        # Print model info
        if 'epoch' in checkpoint:
            print(f"✓ Model trained for {checkpoint['epoch']} epochs")
        
        if 'param_dict' in checkpoint:
            params = checkpoint['param_dict']
            print(f"✓ Model parameters: seq_len={params.get('seq_len', 'unknown')}, "
                  f"bg_mode={params.get('bg_mode', 'unknown')}")
        
        model = model.to(device)
        model.eval()
        
        print(f"✓ Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def preprocess_frames(frames: List[np.ndarray]) -> torch.Tensor:
    """
    Preprocess frames for TrackNetV3 input.
    
    Args:
        frames: List of 9 consecutive frames (RGB)
        
    Returns:
        Preprocessed tensor for model input (1, 27, H, W)
    """
    # Ensure we have exactly 9 frames
    assert len(frames) == 9, f"Expected 9 frames, got {len(frames)}"
    
    # Resize frames
    processed_frames = []
    for frame in frames:
        frame_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        frame_norm = frame_resized.astype(np.float32) / 255.0
        processed_frames.append(frame_norm)
    
    # Stack frames as channels (9 frames × 3 channels = 27 channels)
    input_tensor = np.concatenate(processed_frames, axis=2)  # (H, W, 27)
    input_tensor = np.transpose(input_tensor, (2, 0, 1))     # (27, H, W)
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)  # (1, 27, H, W)
    
    return input_tensor


def postprocess_heatmap(heatmap: torch.Tensor, original_shape: Tuple[int, int], 
                       threshold: float = 0.5) -> Tuple[int, int, float]:
    """
    Postprocess model output heatmap to get ball position.
    
    Args:
        heatmap: Model output heatmap
        original_shape: (height, width) of original video frame
        threshold: Confidence threshold
        
    Returns:
        (x, y, confidence) of ball position
    """
    # Convert to numpy
    heatmap_np = heatmap.squeeze().cpu().numpy()
    
    # Find maximum value and position
    max_val = np.max(heatmap_np)
    max_pos = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
    
    # Scale back to original resolution
    y_scale = original_shape[0] / IMG_HEIGHT
    x_scale = original_shape[1] / IMG_WIDTH
    
    x = int(max_pos[1] * x_scale)
    y = int(max_pos[0] * y_scale)
    
    # Determine visibility (if confidence > threshold)
    visible = 1 if max_val > threshold else 0
    
    return x, y, visible


def run_tracknetv3_inference(video_path: str, output_path: str, 
                           model_path: str = MODEL_PATH,
                           device: str = None) -> str:
    """
    Run TrackNetV3 inference on a video file.
    
    Args:
        video_path: Path to input video
        output_path: Path to save predictions.csv
        model_path: Path to model weights
        device: Device to use ('cpu', 'cuda', or None for auto)
        
    Returns:
        Path to predictions.csv file
    """
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Running TrackNetV3 inference...")
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Output: {output_path}")
    
    # Load model
    model = load_tracknetv3_model(model_path, device)
    if model is None:
        raise RuntimeError("Failed to load model")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
    
    # Prepare results
    results = []
    frame_buffer = []
    
    # Process video frame by frame
    frame_idx = 0
    FRAMES_NEEDED = 9  # Model needs 9 frames for 27 input channels
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame_rgb)
            
            # Keep only the last 9 frames to manage memory
            if len(frame_buffer) > FRAMES_NEEDED:
                frame_buffer = frame_buffer[-FRAMES_NEEDED:]
            
            # Process when we have enough frames
            if len(frame_buffer) >= FRAMES_NEEDED:
                # Use last 9 frames for prediction
                input_frames = frame_buffer[-FRAMES_NEEDED:]
                
                # Preprocess
                input_tensor = preprocess_frames(input_frames).to(device)
                
                # Run inference
                output = model(input_tensor)
                
                # Postprocess
                x, y, visible = postprocess_heatmap(output, (height, width))
                
                # Store result
                results.append({
                    'frame': frame_idx + 1,  # 1-indexed
                    'x': x,
                    'y': y,
                    'vis': visible
                })
                
                # Print progress
                if (frame_idx + 1) % 100 == 0:
                    print(f"Processed {frame_idx + 1}/{total_frames} frames")
            
            frame_idx += 1
    
    cap.release()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Inference complete! Saved {len(results)} predictions to {output_path}")
    print(f"Detected ball in {sum(df['vis'])} frames ({sum(df['vis'])/len(df)*100:.1f}%)")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run TrackNetV3 inference on video")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--model", default=MODEL_PATH, help="Model weights file")
    parser.add_argument("--device", choices=['cpu', 'cuda'], help="Device to use")
    
    args = parser.parse_args()
    
    try:
        run_tracknetv3_inference(args.video, args.output, args.model, args.device)
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 