#!/usr/bin/env python3
"""
Enhanced Badminton Rally Segmentation Pipeline
- TrackNetV3 with M1 acceleration
- Velocity-based rally detection with motion continuity
- Frame quality checks and buffer padding for smooth clips
- Configurable presets for different use cases
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tracknetv3_fast import run_fast_inference
from segment_rallies_enhanced import segment_rallies_enhanced


def run_enhanced_pipeline(video_path: str, 
                         output_dir: str = "clips/enhanced_output",
                         preset: str = "balanced",
                         fps: float = None):
    """
    Run the complete enhanced badminton rally segmentation pipeline
    
    Args:
        video_path: Path to input video
        output_dir: Directory for output rally clips
        preset: Detection preset (conservative|balanced|sensitive)
        fps: Video FPS (auto-detected if None)
    """
    
    print(f"🎾 Enhanced Badminton Rally Segmentation Pipeline")
    print(f"📹 Video: {video_path}")
    print(f"🎯 Output: {output_dir}")
    print(f"⚙️  Preset: {preset}")
    
    # Validate input
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return 1
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/predictions", exist_ok=True)
    
    # Generate prediction filename
    video_name = Path(video_path).stem
    trajectory_path = f"data/predictions/{video_name}_enhanced_predictions.csv"
    
    print(f"\n🧠 Step 1: TrackNetV3 Neural Network Prediction")
    print(f"   Model: TrackNetV3 with M1 MPS acceleration")
    print(f"   Output: {trajectory_path}")
    
    # Step 1: Run TrackNetV3 prediction
    start_time = time.time()
    try:
        run_fast_inference(
            video_path=video_path,
            output_path=trajectory_path,
            model_path="models/tracknetv3/TrackNet_best.pt"
        )
        prediction_time = time.time() - start_time
        print(f"✅ Neural network prediction completed in {prediction_time:.1f}s")
    except Exception as e:
        print(f"❌ Neural network prediction failed: {e}")
        return 1
    
    # Preset configurations
    presets = {
        "conservative": {
            "vel_thresh": 5.0,      # Higher velocity threshold
            "stop_thresh": 2.0,     # Higher stop threshold  
            "occlusion_buffer": 120, # 2 seconds at 60fps
            "min_len": 300,         # 5 seconds minimum
            "stop_duration": 60,    # 1 second stopped
            "description": "Fewer, longer rallies with high confidence"
        },
        "balanced": {
            "vel_thresh": 3.0,      # Medium velocity threshold
            "stop_thresh": 1.0,     # Medium stop threshold
            "occlusion_buffer": 90,  # 1.5 seconds at 60fps
            "min_len": 180,         # 3 seconds minimum
            "stop_duration": 45,    # 0.75 seconds stopped
            "description": "Good balance of accuracy and completeness"
        },
        "sensitive": {
            "vel_thresh": 2.0,      # Lower velocity threshold
            "stop_thresh": 0.5,     # Lower stop threshold
            "occlusion_buffer": 60,  # 1 second at 60fps
            "min_len": 120,         # 2 seconds minimum
            "stop_duration": 30,    # 0.5 seconds stopped
            "description": "More rallies detected, may include short exchanges"
        }
    }
    
    if preset not in presets:
        print(f"❌ Unknown preset: {preset}. Available: {list(presets.keys())}")
        return 1
    
    config = presets[preset]
    
    # Auto-detect FPS if not provided
    if fps is None:
        # Simple heuristic: assume 30fps for smaller files, 60fps for larger
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        fps = 60.0 if file_size_mb > 100 else 30.0
        print(f"🎬 Auto-detected FPS: {fps} (based on file size: {file_size_mb:.1f}MB)")
    
    print(f"\n🔍 Step 2: Enhanced Rally Segmentation")
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
        print(f"✅ Rally segmentation completed in {segmentation_time:.1f}s")
    except Exception as e:
        print(f"❌ Rally segmentation failed: {e}")
        return 1
    
    # Summary
    total_time = prediction_time + segmentation_time
    print(f"\n🎉 Enhanced Pipeline Complete!")
    print(f"   📊 Total time: {total_time:.1f}s")
    print(f"   🏸 Rallies detected: {rally_count}")
    print(f"   💾 Output saved to: {output_dir}")
    
    # Calculate output statistics
    if rally_count > 0:
        total_size_mb = sum(
            os.path.getsize(os.path.join(output_dir, f)) 
            for f in os.listdir(output_dir) 
            if f.endswith('.mp4')
        ) / (1024 * 1024)
        avg_size_mb = total_size_mb / rally_count
        print(f"   📦 Total size: {total_size_mb:.1f}MB, Average: {avg_size_mb:.1f}MB per rally")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced badminton rally segmentation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  conservative  - Fewer, longer rallies with high confidence (recommended for match analysis)
  balanced      - Good balance of accuracy and completeness (default)
  sensitive     - More rallies detected, may include short exchanges (for comprehensive coverage)

Examples:
  python run_enhanced_pipeline.py --video match.mp4
  python run_enhanced_pipeline.py --video match.mp4 --preset conservative --out rally_clips
  python run_enhanced_pipeline.py --video match.mp4 --preset sensitive --fps 30
        """
    )
    
    parser.add_argument("--video", required=True, 
                       help="Input video file path")
    parser.add_argument("--out", default="clips/enhanced_output",
                       help="Output directory for rally clips (default: clips/enhanced_output)")
    parser.add_argument("--preset", choices=["conservative", "balanced", "sensitive"], 
                       default="balanced",
                       help="Detection preset (default: balanced)")
    parser.add_argument("--fps", type=float,
                       help="Video FPS (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    try:
        return run_enhanced_pipeline(
            video_path=args.video,
            output_dir=args.out,
            preset=args.preset,
            fps=args.fps
        )
    except KeyboardInterrupt:
        print(f"\n⚠️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 