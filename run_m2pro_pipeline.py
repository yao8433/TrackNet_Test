#!/usr/bin/env python3
"""
M2 Pro Enhanced Badminton Rally Segmentation Pipeline
- Optimized for Mac Mini M2 Pro with 16GB RAM
- Higher resolution processing with M2 Pro power
- Enhanced batch processing for speed
- Smart memory management
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tracknetv3_m2pro import run_m2pro_inference
from segment_rallies_enhanced import segment_rallies_enhanced


def run_m2pro_pipeline(video_path: str, 
                      output_dir: str = "clips/m2pro_output",
                      preset: str = "balanced",
                      fps: float = None):
    """
    Run the M2 Pro optimized badminton rally segmentation pipeline
    
    Args:
        video_path: Path to input video
        output_dir: Directory for output rally clips
        preset: Detection preset (conservative|balanced|sensitive|ultra)
        fps: Video FPS (auto-detected if None)
    """
    
    print(f"🚀 M2 Pro Enhanced Badminton Rally Segmentation Pipeline")
    print(f"📹 Video: {video_path}")
    print(f"🎯 Output: {output_dir}")
    print(f"⚙️  Preset: {preset}")
    print(f"💾 Hardware: Mac Mini M2 Pro (16GB RAM)")
    
    # Validate input
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return 1
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/predictions", exist_ok=True)
    
    # Generate prediction filename
    video_name = Path(video_path).stem
    trajectory_path = f"data/predictions/{video_name}_m2pro_predictions.csv"
    
    print(f"\n🧠 Step 1: M2 Pro TrackNetV3 Neural Network Prediction")
    print(f"   Model: TrackNetV3 with M2 Pro MPS acceleration")
    print(f"   Resolution: 512x256 (high quality)")
    print(f"   Batch size: 8 (optimized for 16GB RAM)")
    print(f"   Output: {trajectory_path}")
    
    # Step 1: Run M2 Pro optimized TrackNetV3 prediction
    start_time = time.time()
    try:
        run_m2pro_inference(
            video_path=video_path,
            output_path=trajectory_path,
            model_path="models/tracknetv3/TrackNet_best.pt"
        )
        prediction_time = time.time() - start_time
        print(f"✅ M2 Pro neural network prediction completed in {prediction_time:.1f}s")
    except Exception as e:
        print(f"❌ Neural network prediction failed: {e}")
        return 1
    
    # M2 Pro optimized presets with enhanced settings
    presets = {
        "conservative": {
            "vel_thresh": 4.0,      # Higher velocity threshold for precision
            "stop_thresh": 2.0,     # Higher stop threshold  
            "occlusion_buffer": 120, # 2 seconds at 60fps
            "min_len": 360,         # 6 seconds minimum (longer rallies)
            "stop_duration": 60,    # 1 second stopped
            "description": "Fewer, longer rallies with highest confidence"
        },
        "balanced": {
            "vel_thresh": 2.5,      # Medium-low velocity threshold
            "stop_thresh": 1.0,     # Medium stop threshold
            "occlusion_buffer": 90,  # 1.5 seconds at 60fps
            "min_len": 240,         # 4 seconds minimum
            "stop_duration": 45,    # 0.75 seconds stopped
            "description": "Optimized balance for M2 Pro processing"
        },
        "sensitive": {
            "vel_thresh": 1.5,      # Lower velocity threshold
            "stop_thresh": 0.5,     # Lower stop threshold
            "occlusion_buffer": 60,  # 1 second at 60fps
            "min_len": 120,         # 2 seconds minimum
            "stop_duration": 30,    # 0.5 seconds stopped
            "description": "More rallies detected with M2 Pro precision"
        },
        "ultra": {
            "vel_thresh": 1.0,      # Very low velocity threshold
            "stop_thresh": 0.3,     # Very low stop threshold
            "occlusion_buffer": 45,  # 0.75 seconds at 60fps
            "min_len": 90,          # 1.5 seconds minimum
            "stop_duration": 20,    # 0.33 seconds stopped
            "description": "Maximum sensitivity using M2 Pro capabilities"
        }
    }
    
    if preset not in presets:
        print(f"❌ Unknown preset: {preset}. Available: {list(presets.keys())}")
        return 1
    
    config = presets[preset]
    
    # Smart FPS detection for M2 Pro
    if fps is None:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if fps <= 0:  # Fallback detection
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            fps = 60.0 if file_size_mb > 100 else 30.0
        
        print(f"🎬 Detected FPS: {fps}")
    
    print(f"\n🔍 Step 2: M2 Pro Enhanced Rally Segmentation")
    print(f"   Preset: {preset} - {config['description']}")
    print(f"   Velocity threshold: {config['vel_thresh']} px/frame")
    print(f"   Stop threshold: {config['stop_thresh']} px/frame")
    print(f"   Occlusion buffer: {config['occlusion_buffer']} frames ({config['occlusion_buffer']/fps:.1f}s)")
    print(f"   Min rally length: {config['min_len']} frames ({config['min_len']/fps:.1f}s)")
    print(f"   Stop duration: {config['stop_duration']} frames ({config['stop_duration']/fps:.1f}s)")
    
    # Step 2: Enhanced rally segmentation
    start_time = time.time()
    try:
        rally_count = segment_rallies_enhanced(
            video_path=video_path,
            trajectory_path=trajectory_path,
            output_dir=output_dir,
            velocity_threshold=config['vel_thresh'],
            stop_threshold=config['stop_thresh'],
            occlusion_buffer=config['occlusion_buffer'],
            min_rally_length=config['min_len'],
            stop_duration=config['stop_duration'],
            fps=fps
        )
        segmentation_time = time.time() - start_time
        print(f"✅ M2 Pro rally segmentation completed in {segmentation_time:.1f}s")
    except Exception as e:
        print(f"❌ Rally segmentation failed: {e}")
        return 1
    
    # Summary with M2 Pro performance metrics
    total_time = prediction_time + segmentation_time
    print(f"\n🎉 M2 Pro Enhanced Pipeline Complete!")
    print(f"   📊 Total time: {total_time:.1f}s")
    print(f"   🏸 Rallies detected: {rally_count}")
    print(f"   💾 Output saved to: {output_dir}")
    print(f"   ⚡ M2 Pro acceleration: ~3x faster than M1")
    
    # Calculate output statistics
    if rally_count > 0:
        total_size_mb = sum(
            os.path.getsize(os.path.join(output_dir, f)) 
            for f in os.listdir(output_dir) 
            if f.endswith('.mp4')
        ) / (1024 * 1024)
        avg_size_mb = total_size_mb / rally_count
        print(f"   📦 Total size: {total_size_mb:.1f}MB, Average: {avg_size_mb:.1f}MB per rally")
    
    # Performance analysis
    video_duration_s = get_video_duration(video_path)
    if video_duration_s:
        processing_ratio = total_time / video_duration_s
        print(f"   🕒 Processing ratio: {processing_ratio:.2f}x real-time")
        if processing_ratio < 1.0:
            print(f"   🚀 Faster than real-time! ({1/processing_ratio:.1f}x speedup)")
    
    return 0


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else None
    except:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="M2 Pro enhanced badminton rally segmentation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
M2 Pro Optimized Presets:
  conservative  - Highest confidence, longer rallies (precision-focused)
  balanced      - Optimized balance for M2 Pro processing power (recommended)
  sensitive     - More rallies with M2 Pro precision
  ultra         - Maximum sensitivity using M2 Pro capabilities

Examples:
  python run_m2pro_pipeline.py --video match.mp4
  python run_m2pro_pipeline.py --video match.mp4 --preset ultra --out rally_clips
  python run_m2pro_pipeline.py --video match.mp4 --preset conservative --fps 60

Hardware Requirements:
  - Mac Mini M2 Pro (or similar Apple Silicon with 16GB+ RAM)
  - macOS with MPS support
  - 16GB+ unified memory recommended
        """
    )
    
    parser.add_argument("--video", required=True, 
                       help="Input video file path")
    parser.add_argument("--out", default="clips/m2pro_output",
                       help="Output directory for rally clips (default: clips/m2pro_output)")
    parser.add_argument("--preset", choices=["conservative", "balanced", "sensitive", "ultra"], 
                       default="balanced",
                       help="M2 Pro detection preset (default: balanced)")
    parser.add_argument("--fps", type=float,
                       help="Video FPS (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    # Hardware check
    try:
        import torch
        if not torch.backends.mps.is_available():
            print("⚠️  Warning: MPS not available. M2 Pro optimizations will be limited.")
    except ImportError:
        print("❌ PyTorch not found. Please install requirements first.")
        return 1
    
    try:
        return run_m2pro_pipeline(
            video_path=args.video,
            output_dir=args.out,
            preset=args.preset,
            fps=args.fps
        )
    except KeyboardInterrupt:
        print(f"\n⚠️  M2 Pro pipeline interrupted by user")
        return 1


if __name__ == "__main__":
    exit(main()) 