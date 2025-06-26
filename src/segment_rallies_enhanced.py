#!/usr/bin/env python3
"""
Enhanced Rally Segmentation with Velocity-Based Detection
- Velocity-based rally start/end detection
- Occlusion handling with motion continuity
- Noise reduction through movement analysis
- Physics-aware shuttlecock tracking
- Frame quality checks and buffer padding
"""

import argparse
import os
import subprocess
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def calculate_velocity(df: pd.DataFrame, smooth_window: int = 3) -> pd.DataFrame:
    """Calculate shuttlecock velocity with smoothing to reduce noise"""
    
    # Smooth coordinates to reduce tracking noise
    df['x_smooth'] = df['x'].rolling(window=smooth_window, center=True).mean().fillna(df['x'])
    df['y_smooth'] = df['y'].rolling(window=smooth_window, center=True).mean().fillna(df['y'])
    
    # Calculate velocity (pixels per frame)
    df['dx'] = df['x_smooth'].diff()
    df['dy'] = df['y_smooth'].diff()
    
    # Calculate velocity magnitude
    df['velocity'] = np.sqrt(df['dx']**2 + df['dy']**2)
    
    # Fill NaN values
    df['velocity'] = df['velocity'].fillna(0)
    df['dx'] = df['dx'].fillna(0)
    df['dy'] = df['dy'].fillna(0)
    
    return df


def find_quality_start_frame(df: pd.DataFrame, start_frame: int, search_window: int = 30) -> int:
    """
    Find a quality start frame by avoiding black/low-quality frames
    Look for frames with visible shuttlecock near the detected start
    """
    
    # Search window around the detected start
    search_start = max(0, start_frame - search_window)
    search_end = min(len(df), start_frame + search_window)
    
    # Look for frames with good visibility around the start
    search_df = df.iloc[search_start:search_end]
    visible_frames = search_df[search_df['vis'] == 1]
    
    if len(visible_frames) > 0:
        # Prefer frames closer to the original start with good visibility
        best_frame = visible_frames.iloc[0]['frame']
        print(f"   📸 Quality start: moved from frame {start_frame} to {best_frame}")
        return int(best_frame)
    
    # Fallback: use original frame if no visible frames found
    return start_frame


def add_buffer_padding(start_frame: int, end_frame: int, 
                      total_frames: int,
                      start_buffer: int = 60,  # ~2 seconds at 30fps
                      end_buffer: int = 30) -> Tuple[int, int]:   # ~1 second at 30fps
    """
    Add buffer padding before start and after end for smoother transitions
    """
    
    # Add buffer before start (for serve buildup)
    buffered_start = max(1, start_frame - start_buffer)
    
    # Add buffer after end (for shuttlecock landing/completion)
    buffered_end = min(total_frames, end_frame + end_buffer)
    
    start_added = start_frame - buffered_start
    end_added = buffered_end - end_frame
    
    return buffered_start, buffered_end, start_added, end_added


def detect_rally_boundaries(df: pd.DataFrame, 
                          velocity_threshold: float = 2.0,
                          stop_threshold: float = 0.5,
                          occlusion_buffer: int = 60,  # ~2 seconds at 30fps
                          min_rally_length: int = 90,  # ~3 seconds
                          stop_duration: int = 30) -> List[Tuple[int, int]]:
    """
    Enhanced rally boundary detection using velocity and motion continuity
    """
    
    # Calculate velocity
    df = calculate_velocity(df)
    
    print(f"📊 Motion Analysis:")
    print(f"   Max velocity: {df['velocity'].max():.1f} px/frame")
    print(f"   Mean velocity (visible): {df[df['vis']==1]['velocity'].mean():.1f} px/frame")
    print(f"   Velocity threshold: {velocity_threshold} px/frame")
    
    raw_rallies = []
    rally_start = None
    last_motion_frame = None
    invisible_count = 0
    stop_count = 0
    
    for idx, row in df.iterrows():
        frame = row['frame']
        visible = row['vis'] == 1
        velocity = row['velocity']
        in_motion = velocity > velocity_threshold
        stopped = velocity <= stop_threshold
        
        # Rally Start Logic
        if rally_start is None:
            if visible and in_motion:
                rally_start = frame
                last_motion_frame = frame
                invisible_count = 0
                stop_count = 0
                print(f"🚀 Rally start detected at frame {frame} (vel: {velocity:.1f})")
        
        # Rally Continuation Logic  
        else:
            if visible:
                invisible_count = 0
                if in_motion:
                    last_motion_frame = frame
                    stop_count = 0
                elif stopped:
                    stop_count += 1
            else:
                invisible_count += 1
            
            # Rally End Conditions
            rally_ended = False
            end_reason = ""
            
            # 1. Shuttlecock stopped for too long
            if stop_count >= stop_duration:
                rally_ended = True
                end_reason = f"stopped for {stop_count} frames"
            
            # 2. Invisible for too long (occlusion buffer exceeded)
            elif invisible_count >= occlusion_buffer:
                rally_ended = True
                end_reason = f"invisible for {invisible_count} frames"
            
            # 3. End of video
            elif idx == len(df) - 1:
                rally_ended = True
                end_reason = "end of video"
            
            if rally_ended:
                rally_end = last_motion_frame if last_motion_frame else frame
                rally_length = rally_end - rally_start + 1
                
                if rally_length >= min_rally_length:
                    raw_rallies.append((int(rally_start), int(rally_end), end_reason))
                    duration_sec = rally_length / 30  # Assuming 30fps
                    print(f"✅ Raw rally {len(raw_rallies)}: frames {rally_start}-{rally_end} "
                          f"({duration_sec:.1f}s, {end_reason})")
                else:
                    print(f"❌ Rally rejected: {rally_length} frames < {min_rally_length} "
                          f"({end_reason})")
                
                # Reset for next rally
                rally_start = None
                last_motion_frame = None
                invisible_count = 0
                stop_count = 0
    
    # Post-process rallies with quality checks and buffers
    print(f"\n🎬 Post-processing {len(raw_rallies)} raw rallies...")
    processed_rallies = []
    
    for i, (start, end, reason) in enumerate(raw_rallies, 1):
        # 1. Find quality start frame
        quality_start = find_quality_start_frame(df, start, search_window=30)
        
        # 2. Add buffer padding
        buffered_start, buffered_end, start_buffer, end_buffer = add_buffer_padding(
            quality_start, end, len(df), start_buffer=60, end_buffer=30
        )
        
        # 3. Final length check
        final_length = buffered_end - buffered_start + 1
        if final_length >= min_rally_length:
            processed_rallies.append((buffered_start, buffered_end))
            duration_sec = final_length / 30
            print(f"✨ Rally {len(processed_rallies)}: frames {buffered_start}-{buffered_end} "
                  f"({duration_sec:.1f}s, +{start_buffer}f start, +{end_buffer}f end)")
        else:
            print(f"❌ Post-processing rejected rally {i}: {final_length} frames < {min_rally_length}")
    
    return processed_rallies


def analyze_prediction_quality(df: pd.DataFrame) -> dict:
    """Analyze the quality of trajectory predictions"""
    
    visible_frames = df[df['vis'] == 1]
    total_frames = len(df)
    
    # Calculate position variance to detect tracking stability
    x_var = visible_frames['x'].var() if len(visible_frames) > 0 else 0
    y_var = visible_frames['y'].var() if len(visible_frames) > 0 else 0
    
    # Calculate velocity statistics
    df_with_vel = calculate_velocity(df.copy())
    vel_stats = df_with_vel[df_with_vel['vis'] == 1]['velocity'].describe()
    
    # Analyze visibility gaps (potential occlusion patterns)
    df['vis_shift'] = df['vis'].shift(1)
    df['gap_start'] = (df['vis'] == 0) & (df['vis_shift'] == 1)
    df['gap_end'] = (df['vis'] == 1) & (df['vis_shift'] == 0)
    
    gap_starts = df[df['gap_start']]['frame'].tolist()
    gap_ends = df[df['gap_end']]['frame'].tolist()
    
    # Calculate gap lengths
    gap_lengths = []
    for start_idx, start_frame in enumerate(gap_starts):
        if start_idx < len(gap_ends):
            gap_length = gap_ends[start_idx] - start_frame
            gap_lengths.append(gap_length)
    
    return {
        'visibility_rate': len(visible_frames) / total_frames,
        'x_variance': x_var,
        'y_variance': y_var,
        'velocity_stats': vel_stats,
        'total_frames': total_frames,
        'visible_frames': len(visible_frames),
        'gap_count': len(gap_lengths),
        'avg_gap_length': np.mean(gap_lengths) if gap_lengths else 0,
        'max_gap_length': max(gap_lengths) if gap_lengths else 0
    }


def segment_rallies_enhanced(video_path: str, 
                           trajectory_path: str, 
                           output_dir: str,
                           velocity_threshold: float = 2.0,
                           stop_threshold: float = 0.5,
                           occlusion_buffer: int = 60,
                           min_rally_length: int = 90,
                           stop_duration: int = 30,
                           fps: float = 30.0) -> int:
    """Enhanced rally segmentation with velocity-based detection"""
    
    print(f"🎾 Enhanced Rally Segmentation with Quality & Buffers")
    print(f"📹 Video: {video_path}")
    print(f"📊 Trajectory: {trajectory_path}")
    print(f"🎯 Output: {output_dir}")
    
    # Load trajectory data
    df = pd.read_csv(trajectory_path)
    print(f"Loaded {len(df)} trajectory points")
    
    # Analyze prediction quality
    quality = analyze_prediction_quality(df)
    print(f"\n📈 Prediction Quality Analysis:")
    print(f"   Visibility rate: {quality['visibility_rate']*100:.1f}%")
    print(f"   Visible frames: {quality['visible_frames']}/{quality['total_frames']}")
    print(f"   Position variance: x={quality['x_variance']:.1f}, y={quality['y_variance']:.1f}")
    print(f"   Visibility gaps: {quality['gap_count']} gaps, avg={quality['avg_gap_length']:.1f}f, max={quality['max_gap_length']:.1f}f")
    if not quality['velocity_stats'].empty:
        print(f"   Velocity range: {quality['velocity_stats']['min']:.1f} - {quality['velocity_stats']['max']:.1f} px/frame")
        print(f"   Mean velocity: {quality['velocity_stats']['mean']:.1f} px/frame")
    
    # Enhanced rally detection
    print(f"\n🔍 Enhanced Rally Detection:")
    print(f"   Velocity threshold: {velocity_threshold} px/frame")
    print(f"   Stop threshold: {stop_threshold} px/frame") 
    print(f"   Occlusion buffer: {occlusion_buffer} frames ({occlusion_buffer/fps:.1f}s)")
    print(f"   Min rally length: {min_rally_length} frames ({min_rally_length/fps:.1f}s)")
    print(f"   Stop duration: {stop_duration} frames ({stop_duration/fps:.1f}s)")
    print(f"   Start buffer: 60 frames (2.0s), End buffer: 30 frames (1.0s)")
    
    rally_segments = detect_rally_boundaries(
        df=df,
        velocity_threshold=velocity_threshold,
        stop_threshold=stop_threshold,
        occlusion_buffer=occlusion_buffer,
        min_rally_length=min_rally_length,
        stop_duration=stop_duration
    )
    
    if not rally_segments:
        print(f"\n⚠️  No valid rallies detected with current parameters")
        return 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract rally clips
    print(f"\n🎬 Extracting {len(rally_segments)} enhanced rally clips...")
    
    for i, (start_frame, end_frame) in enumerate(rally_segments, 1):
        # Calculate timestamps
        start_time = (start_frame - 1) / fps
        end_time = end_frame / fps
        duration = end_time - start_time
        
        # Generate output filename
        output_file = os.path.join(output_dir, f"rally_{i:03d}_frames_{start_frame}-{end_frame}.mp4")
        
        # FFmpeg command for zero-copy extraction
        cmd = [
            "ffmpeg", "-y", "-v", "quiet",
            "-i", video_path,
            "-ss", f"{start_time:.3f}",
            "-to", f"{end_time:.3f}",
            "-c", "copy",
            output_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"✓ Rally {i}: frames {start_frame}-{end_frame} ({duration:.1f}s, {file_size:.1f}MB)")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to extract rally {i}: {e}")
    
    print(f"\n🎉 Enhanced segmentation complete! {len(rally_segments)} clips saved to {output_dir}")
    
    # Summary statistics
    total_duration = sum((end - start + 1) / fps for start, end in rally_segments)
    avg_duration = total_duration / len(rally_segments)
    print(f"📊 Summary: Total rally time: {total_duration:.1f}s, Average: {avg_duration:.1f}s per rally")
    
    return len(rally_segments)


def main():
    parser = argparse.ArgumentParser(description="Enhanced rally segmentation with quality checks and buffers")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--traj", required=True, help="Trajectory CSV file")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--vel_thresh", type=float, default=2.0, help="Velocity threshold for motion detection (px/frame)")
    parser.add_argument("--stop_thresh", type=float, default=0.5, help="Stop threshold for end detection (px/frame)")
    parser.add_argument("--occlusion_buffer", type=int, default=60, help="Max invisible frames before ending rally")
    parser.add_argument("--min_len", type=int, default=90, help="Minimum rally length in frames")
    parser.add_argument("--stop_duration", type=int, default=30, help="Frames of stopped motion to end rally")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    
    args = parser.parse_args()
    
    try:
        segment_rallies_enhanced(
            video_path=args.video,
            trajectory_path=args.traj,
            output_dir=args.out,
            velocity_threshold=args.vel_thresh,
            stop_threshold=args.stop_thresh,
            occlusion_buffer=args.occlusion_buffer,
            min_rally_length=args.min_len,
            stop_duration=args.stop_duration,
            fps=args.fps
        )
    except Exception as e:
        print(f"❌ Error during enhanced segmentation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 