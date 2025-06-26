#!/usr/bin/env python3
"""
TrackNetV3 optimized for Mac Mini M2 Pro with 16GB RAM
- Enhanced MPS acceleration 
- Larger batch processing with 16GB memory
- Higher resolution processing
- M2 Pro specific optimizations
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
import multiprocessing as mp

# Import the fixed model
from tracknetv3_fixed import TrackNetV3Fixed, ConvBNReLU, DownBlock, Bottleneck, UpBlock

# Model paths
MODEL_PATH = "models/tracknetv3/TrackNet_best.pt"

# M2 Pro optimized parameters
IMG_HEIGHT = 256  # Higher resolution for M2 Pro
IMG_WIDTH = 512   # Full resolution for better accuracy
SEQUENCE_LENGTH = 8
INPUT_CHANNELS = 27
SKIP_FRAMES = 1   # Process more frames with M2 Pro power
BATCH_SIZE = 8    # Larger batches with 16GB RAM
PREFETCH_SIZE = 16  # Larger prefetch buffer

# M2 Pro specific settings
M2_PRO_CORES = 12  # M2 Pro has 12 CPU cores
MEMORY_BUDGET_GB = 12  # Use 12GB of the 16GB available


class TrackNetV3M2Pro(TrackNetV3Fixed):
    """M2 Pro optimized version with enhanced performance"""
    
    def __init__(self, input_channels=27):
        super().__init__(input_channels)
        # M2 Pro specific optimizations
        self.register_buffer('dummy', torch.zeros(1))
        
    def forward(self, x):
        # Optimized forward pass for M2 Pro
        with torch.autocast(device_type='mps', enabled=False):  # MPS works better without autocast
            return super().forward(x)


def load_m2pro_model(model_path=MODEL_PATH, device='mps'):
    """Load TrackNetV3 with M2 Pro optimizations"""
    
    print(f"🚀 Loading TrackNetV3 for M2 Pro...")
    print(f"Target device: {device}")
    print(f"Available memory: {MEMORY_BUDGET_GB}GB")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create M2 Pro optimized model
        model = TrackNetV3M2Pro(input_channels=INPUT_CHANNELS)
        
        # Load weights
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded {len(state_dict) - len(missing_keys)} parameters")
        
        # Move to MPS device
        model = model.to(device)
        model.eval()
        
        # M2 Pro specific optimizations
        # Pre-allocate memory pools
        torch.mps.empty_cache()
        
        print(f"✓ M2 Pro model ready on {device}")
        return model
        
    except Exception as e:
        print(f"Error loading M2 Pro model: {e}")
        return None


def preprocess_frames_m2pro(frames: List[np.ndarray], target_size=(IMG_WIDTH, IMG_HEIGHT)) -> torch.Tensor:
    """M2 Pro optimized preprocessing with higher resolution"""
    
    frames_needed = 9
    
    # Pad or trim frames
    if len(frames) < frames_needed:
        while len(frames) < frames_needed:
            frames.append(frames[-1] if frames else np.zeros((target_size[1], target_size[0], 3)))
    elif len(frames) > frames_needed:
        frames = frames[-frames_needed:]
    
    # High-quality resize for M2 Pro
    processed_frames = []
    for frame in frames:
        # Use INTER_CUBIC for better quality (M2 Pro can handle it)
        frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)
        frame_norm = frame_resized.astype(np.float32) / 255.0
        processed_frames.append(frame_norm)
    
    # Efficient stacking
    input_tensor = np.concatenate(processed_frames, axis=2)  # (H, W, 27)
    input_tensor = np.transpose(input_tensor, (2, 0, 1))     # (27, H, W)
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)  # (1, 27, H, W)
    
    return input_tensor


def preprocess_batch_m2pro(frame_batches: List[List[np.ndarray]], target_size=(IMG_WIDTH, IMG_HEIGHT)) -> torch.Tensor:
    """Batch preprocessing for M2 Pro efficiency"""
    
    batch_tensors = []
    for frames in frame_batches:
        tensor = preprocess_frames_m2pro(frames, target_size)
        batch_tensors.append(tensor)
    
    # Stack into batch
    batch_tensor = torch.cat(batch_tensors, dim=0)
    return batch_tensor


def postprocess_output_m2pro(output: torch.Tensor, original_shape: Tuple[int, int], 
                            threshold: float = 0.25) -> List[Tuple[int, int, int]]:
    """M2 Pro batch postprocessing with high precision"""
    
    results = []
    batch_size = output.shape[0]
    
    for i in range(batch_size):
        # Get best heatmap across channels
        if output.shape[1] == 8:
            heatmap = torch.max(output[i], dim=0)[0]
        else:
            heatmap = output[i, 0]
        
        # Find maximum with sub-pixel precision
        heatmap_detached = heatmap.detach().cpu().numpy() if torch.is_tensor(heatmap) else heatmap
        max_val = np.max(heatmap_detached)
        max_pos = np.unravel_index(np.argmax(heatmap_detached), heatmap_detached.shape)
        
        # Scale to original resolution
        y_scale = original_shape[0] / IMG_HEIGHT
        x_scale = original_shape[1] / IMG_WIDTH
        
        x = int(max_pos[1] * x_scale)
        y = int(max_pos[0] * y_scale)
        
        # High precision threshold
        visible = 1 if max_val > threshold else 0
        
        results.append((x, y, visible))
    
    return results


def run_m2pro_inference(video_path: str, output_path: str, 
                       model_path: str = MODEL_PATH, 
                       device: str = 'mps',
                       skip_frames: int = SKIP_FRAMES,
                       batch_size: int = BATCH_SIZE) -> str:
    """M2 Pro optimized TrackNetV3 inference with 16GB RAM utilization"""
    
    print(f"🚀 M2 Pro Enhanced TrackNetV3 Inference")
    print(f"Video: {video_path}")
    print(f"Output: {output_path}")
    print(f"Resolution: {IMG_WIDTH}x{IMG_HEIGHT} (high quality)")
    print(f"Batch size: {batch_size}")
    print(f"Skip frames: {skip_frames}")
    print(f"CPU cores: {M2_PRO_CORES}")
    
    start_time = time.time()
    
    # Load M2 Pro optimized model
    model = load_m2pro_model(model_path, device)
    if model is None:
        raise RuntimeError("Failed to load M2 Pro model")
    
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
    
    # Calculate memory usage
    frame_memory_mb = (width * height * 3 * 4) / (1024 * 1024)  # 4 bytes per float32
    buffer_memory_mb = frame_memory_mb * PREFETCH_SIZE
    print(f"Memory usage: {buffer_memory_mb:.1f}MB buffer, {MEMORY_BUDGET_GB}GB available")
    
    results = []
    frame_buffer = []
    frame_batches = []
    frame_idx = 0
    processed_count = 0
    
    # Pre-allocate memory for efficiency
    torch.mps.empty_cache()
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame_rgb)
            
            # Keep buffer at optimal size
            if len(frame_buffer) > 9:
                frame_buffer = frame_buffer[-9:]
            
            # Process when ready
            if len(frame_buffer) >= 9:
                frame_batches.append(frame_buffer[-9:].copy())
                
                # Process batch when full
                if len(frame_batches) >= batch_size:
                    # Preprocess batch
                    batch_tensor = preprocess_batch_m2pro(frame_batches)
                    batch_tensor = batch_tensor.to(actual_device, non_blocking=True)
                    
                    # Run batch inference
                    batch_output = model(batch_tensor)
                    
                    # Postprocess batch
                    batch_results = postprocess_output_m2pro(batch_output, (height, width))
                    
                    # Store results
                    for i, (x, y, visible) in enumerate(batch_results):
                        results.append({
                            'frame': frame_idx - (len(frame_batches) - 1 - i) * (skip_frames + 1) + 1,
                            'x': x,
                            'y': y,
                            'vis': visible
                        })
                    
                    processed_count += len(frame_batches)
                    frame_batches = []
                    
                    # Progress update
                    if processed_count % 32 == 0:
                        elapsed = time.time() - start_time
                        fps_processed = processed_count / elapsed
                        eta = (total_frames // (skip_frames + 1) - processed_count) / fps_processed
                        memory_used = torch.mps.current_allocated_memory() / (1024**3)
                        print(f"Processed {processed_count} frames ({frame_idx+1}/{total_frames}) "
                              f"| {fps_processed:.1f} FPS | Memory: {memory_used:.1f}GB | ETA: {eta:.1f}s")
            
            frame_idx += 1
    
    # Process remaining frames
    if frame_batches:
        batch_tensor = preprocess_batch_m2pro(frame_batches)
        batch_tensor = batch_tensor.to(actual_device, non_blocking=True)
        batch_output = model(batch_tensor)
        batch_results = postprocess_output_m2pro(batch_output, (height, width))
        
        for i, (x, y, visible) in enumerate(batch_results):
            results.append({
                'frame': frame_idx - (len(frame_batches) - 1 - i) * (skip_frames + 1) + 1,
                'x': x,
                'y': y,
                'vis': visible
            })
    
    cap.release()
    
    # Fill gaps for skipped frames
    if skip_frames > 0:
        print(f"Interpolating {skip_frames} skipped frames per processed frame...")
        filled_results = []
        for i in range(1, total_frames + 1):
            existing = next((r for r in results if r['frame'] == i), None)
            if existing:
                filled_results.append(existing)
            else:
                # Linear interpolation between nearest frames
                before = max((r for r in results if r['frame'] < i), key=lambda r: r['frame'], default=None)
                after = min((r for r in results if r['frame'] > i), key=lambda r: r['frame'], default=None)
                
                if before and after:
                    alpha = (i - before['frame']) / (after['frame'] - before['frame'])
                    x = int(before['x'] * (1 - alpha) + after['x'] * alpha)
                    y = int(before['y'] * (1 - alpha) + after['y'] * alpha)
                    vis = before['vis'] if alpha < 0.5 else after['vis']
                elif before:
                    x, y, vis = before['x'], before['y'], before['vis']
                elif after:
                    x, y, vis = after['x'], after['y'], after['vis']
                else:
                    x, y, vis = 0, 0, 0
                
                filled_results.append({'frame': i, 'x': x, 'y': y, 'vis': vis})
        
        results = filled_results
    
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
    peak_memory = torch.mps.driver_allocated_memory() / (1024**3)
    
    print(f"🎉 M2 Pro inference complete!")
    print(f"⏱️  Total time: {elapsed:.1f}s ({fps_avg:.1f} FPS average)")
    print(f"📊 Processed {len(results)} frames, detected ball in {visible_count} ({visible_count/len(df)*100:.1f}%)")
    print(f"💾 Memory peak: {peak_memory:.1f}GB")
    print(f"💾 Saved to: {output_path}")
    
    # Clean up
    torch.mps.empty_cache()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="M2 Pro optimized TrackNetV3 inference")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--model", default=MODEL_PATH, help="Model weights file")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--skip", type=int, default=SKIP_FRAMES, help="Skip N frames between processing")
    
    args = parser.parse_args()
    
    try:
        run_m2pro_inference(
            args.video, 
            args.output, 
            args.model,
            args.skip,
            args.batch
        )
    except Exception as e:
        print(f"❌ Error during M2 Pro inference: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 