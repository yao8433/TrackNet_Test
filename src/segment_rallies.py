#!/usr/bin/env python3
"""
Badminton Rally Segmentation with Hybrid Mode

Segments badminton rallies using either:
- vis: Pure visibility-based segmentation (legacy)
- hybrid: Visibility + trajectory direction changes
"""

import argparse
import pandas as pd
import numpy as np
import subprocess
import os
from pathlib import Path
from typing import List, Tuple


def contiguous_ranges(frame_list: List[int]) -> List[Tuple[int, int]]:
    """Convert list of frame numbers to contiguous ranges."""
    if not frame_list:
        return []
    
    frame_list = sorted(frame_list)
    ranges = []
    start = frame_list[0]
    prev = frame_list[0]
    
    for frame in frame_list[1:]:
        if frame != prev + 1:
            ranges.append((start, prev))
            start = frame
        prev = frame
    
    ranges.append((start, prev))
    return ranges


def rally_ranges_from_visibility(df: pd.DataFrame, gap: int = 45, min_len: int = 60) -> List[Tuple[int, int]]:
    """
    Legacy visibility-based rally detection.
    
    Args:
        df: DataFrame with columns [frame, x, y, vis]
        gap: Maximum gap between visible frames to merge
        min_len: Minimum rally length in frames
        
    Returns:
        List of (start_frame, end_frame) tuples
    """
    visible_frames = df[df['vis'] == 1]['frame'].tolist()
    vis_ranges = contiguous_ranges(visible_frames)
    
    # Merge ranges that are close together
    merged_ranges = []
    for start, end in vis_ranges:
        if merged_ranges and start - merged_ranges[-1][1] <= gap:
            # Merge with previous range
            merged_ranges[-1] = (merged_ranges[-1][0], end)
        else:
            merged_ranges.append((start, end))
    
    # Filter by minimum length
    return [(start, end) for start, end in merged_ranges if end - start >= min_len]


def detect_hits_from_trajectory(df: pd.DataFrame, gap_hit: int = 120, min_len: int = 60) -> List[Tuple[int, int]]:
    """
    Detect rally ranges based on y-velocity direction changes (hits).
    
    Args:
        df: DataFrame with columns [frame, x, y, vis]
        gap_hit: Maximum gap between hits to consider same rally
        min_len: Minimum rally length in frames
        
    Returns:
        List of (start_frame, end_frame) tuples based on hit clustering
    """
    # Calculate y-velocity (dy/dt)
    df_work = df.copy()
    df_work['dy'] = df_work['y'].diff().fillna(0)
    
    # Detect direction changes: where dy changes sign and shuttle is visible
    # This indicates potential hits
    sign_changes = []
    for i in range(1, len(df_work)):
        if (df_work.iloc[i]['vis'] == 1 and 
            df_work.iloc[i-1]['vis'] == 1 and
            df_work.iloc[i]['dy'] * df_work.iloc[i-1]['dy'] < 0 and
            abs(df_work.iloc[i]['dy']) > 1):  # Minimum velocity change threshold
            sign_changes.append(df_work.iloc[i]['frame'])
    
    if not sign_changes:
        return []
    
    # Cluster hits into rallies based on gap_hit threshold
    rallies = []
    rally_start = sign_changes[0]
    rally_hits = [sign_changes[0]]
    
    for hit_frame in sign_changes[1:]:
        if hit_frame - rally_hits[-1] <= gap_hit:
            # Continue current rally
            rally_hits.append(hit_frame)
        else:
            # End current rally and start new one
            if len(rally_hits) >= 2:  # Need at least 2 hits for a rally
                rallies.append((rally_start, rally_hits[-1]))
            rally_start = hit_frame
            rally_hits = [hit_frame]
    
    # Don't forget the last rally
    if len(rally_hits) >= 2:
        rallies.append((rally_start, rally_hits[-1]))
    
    # Filter by minimum length
    return [(start, end) for start, end in rallies if end - start >= min_len]


def expand_hits_to_visibility_spans(hit_ranges: List[Tuple[int, int]], 
                                  vis_spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Expand hit-based rally ranges to full visibility spans.
    
    Args:
        hit_ranges: List of (first_hit, last_hit) tuples
        vis_spans: List of contiguous visibility spans
        
    Returns:
        List of expanded (start_frame, end_frame) tuples
    """
    expanded_ranges = []
    
    for hit_start, hit_end in hit_ranges:
        # Find visibility span containing hit_start
        start_span = None
        for vs_start, vs_end in vis_spans:
            if vs_start <= hit_start <= vs_end:
                start_span = (vs_start, vs_end)
                break
        
        # Find visibility span containing hit_end
        end_span = None
        for vs_start, vs_end in vis_spans:
            if vs_start <= hit_end <= vs_end:
                end_span = (vs_start, vs_end)
                break
        
        if start_span and end_span:
            # Expand from start of first span to end of last span
            expanded_start = start_span[0]
            expanded_end = end_span[1]
            expanded_ranges.append((expanded_start, expanded_end))
    
    return expanded_ranges


def extract_video_clip(video_path: str, start_frame: int, end_frame: int, 
                      output_path: str, fps: float = 30.0):
    """
    Extract video clip using ffmpeg with frame-accurate cutting.
    
    Args:
        video_path: Input video file path
        start_frame: Starting frame number
        end_frame: Ending frame number
        output_path: Output clip file path
        fps: Video frame rate
    """
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps
    
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', video_path,
        '-t', str(duration),
        '-c', 'copy',  # No re-encoding for speed
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Extracted clip: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to extract clip: {e}")


def main():
    parser = argparse.ArgumentParser(description="Segment badminton rallies from trajectory data")
    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument("--traj", required=True, help="Trajectory CSV file (frame,x,y,vis)")
    parser.add_argument("--out", required=True, help="Output directory for clips")
    parser.add_argument("--mode", choices=["vis", "hybrid"], default="hybrid",
                       help="Segmentation mode: vis (visibility only) or hybrid (visibility + trajectory)")
    parser.add_argument("--gap", type=int, default=45,
                       help="Max frame gap for visibility mode")
    parser.add_argument("--gap_hit", type=int, default=120,
                       help="Max frame gap between hits for hybrid mode")
    parser.add_argument("--min_len", type=int, default=60,
                       help="Minimum rally length in frames")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="Video frame rate")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Load trajectory data
    print(f"Loading trajectory data from {args.traj}")
    df = pd.read_csv(args.traj)
    
    # Validate DataFrame structure
    required_cols = ['frame', 'x', 'y', 'vis']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    print(f"Loaded {len(df)} trajectory points")
    
    # Segment rallies based on mode
    if args.mode == "hybrid":
        print("Using hybrid mode (visibility + trajectory)")
        
        # 1. Detect hit-frames by y-velocity direction changes
        hit_ranges = detect_hits_from_trajectory(df, args.gap_hit, args.min_len)
        print(f"Detected {len(hit_ranges)} rally candidates from trajectory analysis")
        
        if hit_ranges:
            # 2. Get visibility spans
            visible_frames = df[df['vis'] == 1]['frame'].tolist()
            vis_spans = contiguous_ranges(visible_frames)
            
            # 3. Expand hit ranges to full visibility spans
            spans = expand_hits_to_visibility_spans(hit_ranges, vis_spans)
        else:
            spans = []
            
    else:  # mode == "vis"
        print("Using visibility mode (legacy)")
        spans = rally_ranges_from_visibility(df, args.gap, args.min_len)
    
    print(f"Final rally segments: {len(spans)}")
    
    # Extract video clips
    for i, (start_frame, end_frame) in enumerate(spans, 1):
        duration_sec = (end_frame - start_frame) / args.fps
        output_filename = f"rally_{i:03d}_frames_{start_frame}-{end_frame}.mp4"
        output_path = os.path.join(args.out, output_filename)
        
        print(f"Rally {i}: frames {start_frame}-{end_frame} ({duration_sec:.1f}s)")
        extract_video_clip(args.video, start_frame, end_frame, output_path, args.fps)
    
    print(f"\n✓ Segmentation complete! {len(spans)} clips saved to {args.out}")


if __name__ == "__main__":
    main() 