#!/usr/bin/env python3
"""
Fixed TrackNetV3 inference with correct architecture matching the checkpoint
Based on the actual model structure from model inspection
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from typing import Tuple, List

# Model paths
MODEL_PATH = "models/tracknetv3/TrackNet_best.pt"

# Video processing parameters (from checkpoint param_dict)
IMG_HEIGHT = 288
IMG_WIDTH = 512
SEQUENCE_LENGTH = 8  # seq_len from param_dict
INPUT_CHANNELS = 27  # 9 frames × 3 RGB channels


class ConvBNReLU(nn.Module):
    """Conv + BatchNorm + ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DownBlock(nn.Module):
    """Down sampling block matching the checkpoint structure"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv_1 = ConvBNReLU(in_channels, out_channels)
        self.conv_2 = ConvBNReLU(out_channels, out_channels)
        # Some blocks have conv_3 based on inspection
        self.conv_3 = None
        if out_channels == 256:  # down_block_3 has conv_3
            self.conv_3 = ConvBNReLU(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        if self.conv_3 is not None:
            x = self.conv_3(x)
        skip = x
        x = self.pool(x)
        return x, skip


class Bottleneck(nn.Module):
    """Bottleneck layer matching checkpoint structure"""
    def __init__(self, in_channels):
        super(Bottleneck, self).__init__()
        self.conv_1 = ConvBNReLU(in_channels, 512)
        self.conv_2 = ConvBNReLU(512, 512)
        self.conv_3 = ConvBNReLU(512, 512)
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


class UpBlock(nn.Module):
    """Up sampling block using interpolation (not transpose conv)"""
    def __init__(self, in_channels, out_channels, has_conv3=False):
        super(UpBlock, self).__init__()
        # No transpose conv - just interpolation upsampling
        # The input channels for conv_1 include both upsampled and skip connection
        # From checkpoint analysis:
        # up_block_1: 768 channels = 512 (from bottleneck) + 256 (skip from down_block_3)
        # up_block_2: 384 channels = 256 (from up_block_1) + 128 (skip from down_block_2)  
        # up_block_3: 192 channels = 128 (from up_block_2) + 64 (skip from down_block_1)
        
        skip_channels_map = {
            256: 256,  # down_block_3 output
            128: 128,  # down_block_2 output
            64: 64     # down_block_1 output
        }
        conv_in_channels = in_channels + skip_channels_map.get(out_channels, out_channels)
        
        self.conv_1 = ConvBNReLU(conv_in_channels, out_channels)
        self.conv_2 = ConvBNReLU(out_channels, out_channels)
        
        # up_block_1 has conv_3
        self.conv_3 = None
        if has_conv3:
            self.conv_3 = ConvBNReLU(out_channels, out_channels)
    
    def forward(self, x, skip):
        # Upsample using interpolation to match skip connection size
        x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Apply convolutions
        x = self.conv_1(x)
        x = self.conv_2(x)
        if self.conv_3 is not None:
            x = self.conv_3(x)
        return x


class TrackNetV3Fixed(nn.Module):
    """
    TrackNetV3 with architecture exactly matching the checkpoint
    """
    def __init__(self, input_channels=27):
        super(TrackNetV3Fixed, self).__init__()
        
        # Down blocks (encoder)
        self.down_block_1 = DownBlock(input_channels, 64)
        self.down_block_2 = DownBlock(64, 128)
        self.down_block_3 = DownBlock(128, 256)  # This one has conv_3
        
        # Bottleneck
        self.bottleneck = Bottleneck(256)
        
        # Up blocks (decoder)
        self.up_block_1 = UpBlock(512, 256, has_conv3=True)  # This one has conv_3
        self.up_block_2 = UpBlock(256, 128)
        self.up_block_3 = UpBlock(128, 64)
        
        # Final prediction layer - from checkpoint: (8, 64, 1, 1)
        self.predictor = nn.Conv2d(64, 8, kernel_size=1, padding=0)  # 8 channels, 1x1 kernel
        
    def forward(self, x):
        # Encoder path
        x1, skip1 = self.down_block_1(x)
        x2, skip2 = self.down_block_2(x1)
        x3, skip3 = self.down_block_3(x2)
        
        # Bottleneck
        x = self.bottleneck(x3)
        
        # Decoder path with skip connections
        x = self.up_block_1(x, skip3)
        x = self.up_block_2(x, skip2)
        x = self.up_block_3(x, skip1)
        
        # Final prediction
        x = self.predictor(x)
        return x


def load_fixed_model(model_path=MODEL_PATH, device='cpu'):
    """Load the TrackNetV3 model with correct architecture"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading TrackNetV3 model from: {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model
        model = TrackNetV3Fixed(input_channels=INPUT_CHANNELS)
        
        # Load state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            
            # Try to load with strict=False to see what matches
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
                for key in missing_keys[:5]:  # Show first 5
                    print(f"  - {key}")
                if len(missing_keys) > 5:
                    print(f"  ... and {len(missing_keys)-5} more")
            
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
                for key in unexpected_keys[:5]:  # Show first 5
                    print(f"  - {key}")
                if len(unexpected_keys) > 5:
                    print(f"  ... and {len(unexpected_keys)-5} more")
            
            print(f"✓ Model loaded with {len(state_dict) - len(missing_keys)} matching parameters")
        
        # Print model info
        if 'epoch' in checkpoint:
            print(f"✓ Model trained for {checkpoint['epoch']} epochs")
        
        if 'param_dict' in checkpoint:
            params = checkpoint['param_dict']
            print(f"✓ Model parameters: seq_len={params.get('seq_len')}, bg_mode={params.get('bg_mode')}")
        
        model = model.to(device)
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def preprocess_frames_fixed(frames: List[np.ndarray]) -> torch.Tensor:
    """Preprocess frames for TrackNetV3 input (9 frames)"""
    
    # Need exactly 9 frames for 27 channels
    frames_needed = 9
    
    # Pad or trim frames
    if len(frames) < frames_needed:
        while len(frames) < frames_needed:
            frames.append(frames[-1] if frames else np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)))
    elif len(frames) > frames_needed:
        frames = frames[-frames_needed:]
    
    # Resize and normalize
    processed_frames = []
    for frame in frames:
        frame_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        frame_norm = frame_resized.astype(np.float32) / 255.0
        processed_frames.append(frame_norm)
    
    # Stack as channels: 9 frames × 3 channels = 27 channels
    input_tensor = np.concatenate(processed_frames, axis=2)  # (H, W, 27)
    input_tensor = np.transpose(input_tensor, (2, 0, 1))     # (27, H, W)
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)  # (1, 27, H, W)
    
    return input_tensor


def postprocess_output_fixed(output: torch.Tensor, original_shape: Tuple[int, int], 
                           threshold: float = 0.5) -> Tuple[int, int, int]:
    """Postprocess model output to get ball position"""
    
    # Output is (1, 8, H, W) - typically channel 0 is the ball heatmap
    # or we can use the maximum across all channels
    if output.shape[1] == 8:
        # Take maximum across all 8 channels to get best heatmap
        heatmap = torch.max(output[0], dim=0)[0].cpu().numpy()
    else:
        # Fallback to first channel
        heatmap = output[0, 0].cpu().numpy()
    
    # Find maximum value and position
    max_val = np.max(heatmap)
    max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    # Scale back to original resolution
    y_scale = original_shape[0] / IMG_HEIGHT
    x_scale = original_shape[1] / IMG_WIDTH
    
    x = int(max_pos[1] * x_scale)
    y = int(max_pos[0] * y_scale)
    
    # Determine visibility
    visible = 1 if max_val > threshold else 0
    
    return x, y, visible


def run_fixed_inference(video_path: str, output_path: str, 
                       model_path: str = MODEL_PATH, device: str = None) -> str:
    """Run TrackNetV3 inference with fixed architecture"""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Running Fixed TrackNetV3 inference...")
    print(f"Video: {video_path}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")
    
    # Load model
    model = load_fixed_model(model_path, device)
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
    
    results = []
    frame_buffer = []
    frame_idx = 0
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame_rgb)
            
            # Keep buffer size manageable
            if len(frame_buffer) > 9:
                frame_buffer = frame_buffer[-9:]
            
            # Process when we have enough frames
            if len(frame_buffer) >= 9:
                # Preprocess
                input_tensor = preprocess_frames_fixed(frame_buffer[-9:]).to(device)
                
                # Run inference
                output = model(input_tensor)
                
                # Postprocess
                x, y, visible = postprocess_output_fixed(output, (height, width))
                
                # Store result
                results.append({
                    'frame': frame_idx + 1,
                    'x': x,
                    'y': y,
                    'vis': visible
                })
                
                # Print progress
                if (frame_idx + 1) % 200 == 0:
                    print(f"Processed {frame_idx + 1}/{total_frames} frames")
            
            frame_idx += 1
    
    cap.release()
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    visible_count = sum(df['vis'])
    print(f"✓ Inference complete! Saved {len(results)} predictions to {output_path}")
    print(f"✓ Detected ball in {visible_count} frames ({visible_count/len(df)*100:.1f}%)")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Fixed TrackNetV3 inference")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--model", default=MODEL_PATH, help="Model weights file")
    parser.add_argument("--device", choices=['cpu', 'cuda'], help="Device to use")
    
    args = parser.parse_args()
    
    try:
        run_fixed_inference(args.video, args.output, args.model, args.device)
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 