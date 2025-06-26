#!/usr/bin/env python3
"""
Calibrated Badminton Rally Segmentation Pipeline
Optimized based on 5-rally test video analysis
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
    parser = argparse.ArgumentParser(description="Calibrated badminton rally segmentation pipeline")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--output_dir", default="clips/calibrated_clips", help="Output directory for clips")
    parser.add_argument("--device", choices=['cpu', 'mps', 'cuda', 'auto'], default='auto', help="Device")
    parser.add_argument("--profile", choices=['conservative', 'balanced', 'sensitive'], default='balanced', 
                       help="Detection profile")
    
    args = parser.parse_args()
    
    # Calibrated profiles based on 5-rally test analysis
    profiles = {
        'conservative': {
            'skip': 3, 'gap': 60, 'min_len': 90, 'mode': 'vis',
            'description': '~4 rallies, longer segments, fewer false positives'
        },
        'balanced': {
            'skip': 3, 'gap': 45, 'min_len': 75, 'mode': 'vis', 
            'description': '~5-6 rallies, good balance of accuracy vs completeness'
        },
        'sensitive': {
            'skip': 3, 'gap': 30, 'min_len': 45, 'mode': 'vis',
            'description': '~8 rallies, catches short exchanges, may over-segment'
        }
    }
    
    profile = profiles[args.profile]
    
    # Derived paths
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    predictions_file = f"data/predictions/{video_name}_calibrated_predictions.csv"
    
    print(f"🏸 Calibrated Badminton Rally Segmentation Pipeline")
    print(f"📹 Video: {args.video}")
    print(f"🎯 Output: {args.output_dir}")
    print(f"⚡ Device: {args.device}")
    print(f"📊 Profile: {args.profile} - {profile['description']}")
    print(f"🔧 Parameters: gap={profile['gap']}, min_len={profile['min_len']}, skip={profile['skip']}")
    
    # Step 1: Fast neural network inference
    step1_cmd = [
        "python", "src/tracknetv3_fast.py",
        "--video", args.video,
        "--output", predictions_file,
        "--device", args.device,
        "--skip", str(profile['skip'])
    ]
    
    if not run_command(step1_cmd, "Neural Network Inference (TrackNetV3)"):
        return 1
    
    # Step 2: Rally segmentation with calibrated parameters  
    step2_cmd = [
        "python", "src/segment_rallies.py",
        "--video", args.video,
        "--traj", predictions_file,
        "--out", args.output_dir,
        "--mode", profile['mode'],
        "--gap", str(profile['gap']),
        "--min_len", str(profile['min_len']),
        "--fps", "30"
    ]
    
    if not run_command(step2_cmd, "Rally Segmentation"):
        return 2
    
    # Summary with calibration info
    print(f"\n🎉 Calibrated Pipeline Complete!")
    print(f"📊 Check your rally clips in: {args.output_dir}")
    print(f"📈 Predictions saved to: {predictions_file}")
    
    # Show clips if they exist
    if os.path.exists(args.output_dir):
        clips = [f for f in os.listdir(args.output_dir) if f.endswith('.mp4')]
        if clips:
            print(f"\n🎬 Generated {len(clips)} rally clips:")
            total_size = 0
            for clip in sorted(clips):
                clip_path = os.path.join(args.output_dir, clip)
                size_mb = os.path.getsize(clip_path) / (1024*1024)
                total_size += size_mb
                print(f"  • {clip} ({size_mb:.1f} MB)")
            print(f"📦 Total size: {total_size:.1f} MB")
            
            # Calibration feedback
            expected_range = {
                'conservative': '3-5',
                'balanced': '5-7', 
                'sensitive': '7-10'
            }
            expected = expected_range[args.profile]
            print(f"\n📈 Calibration Analysis:")
            print(f"   Expected rallies: {expected}")
            print(f"   Detected rallies: {len(clips)}")
            
            if len(clips) < int(expected.split('-')[0]):
                print(f"   💡 Tip: Try 'sensitive' profile for more rallies")
            elif len(clips) > int(expected.split('-')[1]):
                print(f"   💡 Tip: Try 'conservative' profile for fewer, longer rallies")
            else:
                print(f"   ✅ Detection count within expected range!")
                
        else:
            print(f"\n⚠️  No rally clips generated")
            print(f"   💡 Try 'sensitive' profile or check video content")
    
    return 0


if __name__ == "__main__":
    exit(main()) 