#!/usr/bin/env python3
"""
Fast TrackNetV3 inference optimized for M1 MacBook
- MPS acceleration for Apple Silicon
- Lower resolution processing
- Batch processing
- Skip frame optimizations
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from typing import Tuple, List
import time

# Import the fixed model
from tracknetv3_fixed import TrackNetV3Fixed, ConvBNReLU, DownBlock, Bottleneck, UpBlock

# Model paths
MODEL_PATH = "models/tracknetv3/TrackNet_best.pt"

# Optimized parameters for speed
IMG_HEIGHT = 144  # Half resolution: 288 -> 144
IMG_WIDTH = 256   # Half resolution: 512 -> 256
SEQUENCE_LENGTH = 8
INPUT_CHANNELS = 27
SKIP_FRAMES = 2   # Process every 2nd frame for speed
BATCH_SIZE = 4    # Process multiple frames at once


class TrackNetV3Fast(TrackNetV3Fixed):
    """Fast version with optimizations"""
    def __init__(self, input_channels=27):
        super().__init__(input_channels)
        
    def forward(self, x):
        # Add some optimizations
        with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for MPS compatibility
            return super().forward(x)


def load_fast_model(model_path=MODEL_PATH, device='mps'):
    """Load TrackNetV3 with optimizations for M1"""
    
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    print(f"Loading Fast TrackNetV3 model...")
    print(f"Target device: {device}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create fast model
        model = TrackNetV3Fast(input_channels=INPUT_CHANNELS)
        
        # Load weights
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded {len(state_dict) - len(missing_keys)} parameters")
        
        # Move to device and optimize
        model = model.to(device)
        model.eval()
        
        # Compile for extra speed (if available)
        try:
            if hasattr(torch, 'compile') and device != 'mps':  # MPS doesn't support compile yet
                model = torch.compile(model)
                print("✓ Model compiled for extra speed")
        except:
            pass
            
        print(f"✓ Model ready on {device}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def preprocess_frames_fast(frames: List[np.ndarray], target_size=(IMG_WIDTH, IMG_HEIGHT)) -> torch.Tensor:
    """Fast preprocessing with lower resolution"""
    
    frames_needed = 9
    
    # Pad or trim frames
    if len(frames) < frames_needed:
        while len(frames) < frames_needed:
            frames.append(frames[-1] if frames else np.zeros((target_size[1], target_size[0], 3)))
    elif len(frames) > frames_needed:
        frames = frames[-frames_needed:]
    
    # Fast resize and normalize
    processed_frames = []
    for frame in frames:
        # Use INTER_LINEAR for speed
        frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        frame_norm = frame_resized.astype(np.float32) / 255.0
        processed_frames.append(frame_norm)
    
    # Stack efficiently
    input_tensor = np.concatenate(processed_frames, axis=2)  # (H, W, 27)
    input_tensor = np.transpose(input_tensor, (2, 0, 1))     # (27, H, W)
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)  # (1, 27, H, W)
    
    return input_tensor


def postprocess_output_fast(output: torch.Tensor, original_shape: Tuple[int, int], 
                          threshold: float = 0.3) -> Tuple[int, int, int]:
    """Fast postprocessing with lower threshold"""
    
    # Get best heatmap across channels
    if output.shape[1] == 8:
        heatmap = torch.max(output[0], dim=0)[0].cpu().numpy()
    else:
        heatmap = output[0, 0].cpu().numpy()
    
    # Find maximum
    max_val = np.max(heatmap)
    max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    # Scale to original resolution
    y_scale = original_shape[0] / IMG_HEIGHT
    x_scale = original_shape[1] / IMG_WIDTH
    
    x = int(max_pos[1] * x_scale)
    y = int(max_pos[0] * y_scale)
    
    # More lenient threshold for speed
    visible = 1 if max_val > threshold else 0
    
    return x, y, visible


def run_fast_inference(video_path: str, output_path: str, 
                      model_path: str = MODEL_PATH, 
                      device: str = 'auto',
                      skip_frames: int = SKIP_FRAMES,
                      fast_mode: bool = True) -> str:
    """Fast TrackNetV3 inference with M1 optimizations"""
    
    print(f"🚀 Running FAST TrackNetV3 inference...")
    print(f"Video: {video_path}")
    print(f"Output: {output_path}")
    print(f"Skip frames: {skip_frames} (process every {skip_frames+1} frames)")
    print(f"Resolution: {IMG_WIDTH}x{IMG_HEIGHT} (fast mode: {fast_mode})")
    
    start_time = time.time()
    
    # Load model
    model = load_fast_model(model_path, device)
    if model is None:
        raise RuntimeError("Failed to load model")
    
    actual_device = next(model.parameters()).device
    print(f"Model loaded on: {actual_device}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")
    
    results = []
    frame_buffer = []
    frame_idx = 0
    processed_count = 0
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for speed
            if frame_idx % (skip_frames + 1) != 0:
                # Still need to fill gaps in results for skipped frames
                if results:  # Use last prediction
                    last_result = results[-1].copy()
                    last_result['frame'] = frame_idx + 1
                    results.append(last_result)
                frame_idx += 1
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame_rgb)
            
            # Keep buffer manageable
            if len(frame_buffer) > 9:
                frame_buffer = frame_buffer[-9:]
            
            # Process when ready
            if len(frame_buffer) >= 9:
                # Preprocess
                input_tensor = preprocess_frames_fast(frame_buffer[-9:])
                input_tensor = input_tensor.to(actual_device, non_blocking=True)
                
                # Run inference
                output = model(input_tensor)
                
                # Postprocess
                x, y, visible = postprocess_output_fast(output, (height, width))
                
                # Store result
                results.append({
                    'frame': frame_idx + 1,
                    'x': x,
                    'y': y,
                    'vis': visible
                })
                
                processed_count += 1
                
                # Progress update
                if processed_count % 50 == 0:
                    elapsed = time.time() - start_time
                    fps_processed = processed_count / elapsed
                    eta = (total_frames // (skip_frames + 1) - processed_count) / fps_processed
                    print(f"Processed {processed_count} frames ({frame_idx+1}/{total_frames}) "
                          f"| {fps_processed:.1f} FPS | ETA: {eta:.1f}s")
            
            frame_idx += 1
    
    cap.release()
    
    # Fill any remaining gaps
    if len(results) < total_frames:
        print(f"Filling {total_frames - len(results)} gap frames...")
        for i in range(1, total_frames + 1):
            if not any(r['frame'] == i for r in results):
                # Find nearest prediction
                nearest = min(results, key=lambda r: abs(r['frame'] - i))
                gap_result = nearest.copy()
                gap_result['frame'] = i
                results.append(gap_result)
    
    # Sort by frame
    results.sort(key=lambda x: x['frame'])
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    # Statistics
    elapsed = time.time() - start_time
    visible_count = sum(df['vis'])
    fps_avg = len(results) / elapsed
    
    print(f"🎉 Fast inference complete!")
    print(f"⏱️  Total time: {elapsed:.1f}s ({fps_avg:.1f} FPS average)")
    print(f"📊 Processed {len(results)} frames, detected ball in {visible_count} ({visible_count/len(df)*100:.1f}%)")
    print(f"💾 Saved to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Fast TrackNetV3 inference for M1")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--model", default=MODEL_PATH, help="Model weights file")
    parser.add_argument("--device", choices=['cpu', 'mps', 'cuda', 'auto'], default='auto', help="Device")
    parser.add_argument("--skip", type=int, default=SKIP_FRAMES, help="Skip N frames between processing")
    parser.add_argument("--no-fast", action='store_true', help="Disable fast optimizations")
    
    args = parser.parse_args()
    
    try:
        run_fast_inference(
            args.video, 
            args.output, 
            args.model, 
            args.device,
            args.skip,
            not args.no_fast
        )
    except Exception as e:
        print(f"❌ Error during fast inference: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 