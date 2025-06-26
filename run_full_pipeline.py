#!/usr/bin/env python3
"""
Complete Badminton Rally Segmentation Pipeline
Optimized for M1 MacBook with fast TrackNetV3 + Rally Segmentation
"""

import argparse
import os
import time
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and show progress"""
    print(f"\n🚀 {description}...")
    print(f"Command: {' '.join(cmd)}")
    start_time = time.time()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ {description} completed in {elapsed:.1f}s")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ {description} failed after {elapsed:.1f}s")
        print(f"Error: {result.stderr}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Complete badminton rally segmentation pipeline")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--output_dir", default="clips/auto_clips", help="Output directory for clips")
    parser.add_argument("--device", choices=['cpu', 'mps', 'cuda', 'auto'], default='auto', help="Device for neural network")
    parser.add_argument("--skip", type=int, default=3, help="Skip N frames for speed (0=no skip)")
    parser.add_argument("--mode", choices=['vis', 'hybrid'], default='vis', help="Segmentation mode")
    parser.add_argument("--gap", type=int, default=60, help="Gap frames for visibility mode")
    parser.add_argument("--gap_hit", type=int, default=120, help="Gap frames for hybrid hit detection")
    parser.add_argument("--min_len", type=int, default=60, help="Minimum rally length in frames")
    
    args = parser.parse_args()
    
    # Derived paths
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    predictions_file = f"data/predictions/{video_name}_fast_predictions.csv"
    
    print(f"🏸 Complete Badminton Rally Segmentation Pipeline")
    print(f"📹 Video: {args.video}")
    print(f"🎯 Output: {args.output_dir}")
    print(f"⚡ Device: {args.device}")
    print(f"🏃 Skip frames: {args.skip}")
    print(f"🧠 Mode: {args.mode}")
    
    # Step 1: Fast neural network inference
    step1_cmd = [
        "python", "src/tracknetv3_fast.py",
        "--video", args.video,
        "--output", predictions_file,
        "--device", args.device,
        "--skip", str(args.skip)
    ]
    
    if not run_command(step1_cmd, "Neural Network Inference (TrackNetV3)"):
        return 1
    
    # Step 2: Rally segmentation
    step2_cmd = [
        "python", "src/segment_rallies.py",
        "--video", args.video,
        "--traj", predictions_file,
        "--out", args.output_dir,
        "--mode", args.mode,
        "--min_len", str(args.min_len),
        "--fps", "30"
    ]
    
    if args.mode == "vis":
        step2_cmd.extend(["--gap", str(args.gap)])
    else:
        step2_cmd.extend(["--gap_hit", str(args.gap_hit)])
    
    if not run_command(step2_cmd, "Rally Segmentation"):
        return 2
    
    # Summary
    print(f"\n🎉 Pipeline Complete!")
    print(f"📊 Check your rally clips in: {args.output_dir}")
    print(f"📈 Predictions saved to: {predictions_file}")
    
    # Show clips if they exist
    if os.path.exists(args.output_dir):
        clips = [f for f in os.listdir(args.output_dir) if f.endswith('.mp4')]
        if clips:
            print(f"\n🎬 Generated {len(clips)} rally clips:")
            for clip in sorted(clips):
                clip_path = os.path.join(args.output_dir, clip)
                size_mb = os.path.getsize(clip_path) / (1024*1024)
                print(f"  • {clip} ({size_mb:.1f} MB)")
        else:
            print(f"\n⚠️  No rally clips generated - try adjusting parameters")
    
    return 0


if __name__ == "__main__":
    exit(main()) 