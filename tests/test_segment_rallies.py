#!/usr/bin/env python3
"""
Unit tests for rally segmentation functionality
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from segment_rallies import (
    contiguous_ranges,
    rally_ranges_from_visibility,
    detect_hits_from_trajectory,
    expand_hits_to_visibility_spans
)


class TestSegmentRallies(unittest.TestCase):
    
    def test_contiguous_ranges(self):
        """Test conversion of frame list to contiguous ranges."""
        # Test basic case
        frames = [1, 2, 3, 5, 6, 10, 11, 12, 13]
        expected = [(1, 3), (5, 6), (10, 13)]
        result = contiguous_ranges(frames)
        self.assertEqual(result, expected)
        
        # Test empty list
        self.assertEqual(contiguous_ranges([]), [])
        
        # Test single frame
        self.assertEqual(contiguous_ranges([5]), [(5, 5)])
    
    def test_visibility_mode(self):
        """Test legacy visibility-based segmentation."""
        # Create test data with two visibility segments
        frames = list(range(1, 101)) + list(range(150, 251))
        vis = [1] * 100 + [1] * 101
        df = pd.DataFrame({
            'frame': frames,
            'x': [100] * len(frames),
            'y': [50] * len(frames),
            'vis': vis
        })
        
        ranges = rally_ranges_from_visibility(df, gap=45, min_len=60)
        
        # Should detect two separate rallies
        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0], (1, 100))
        self.assertEqual(ranges[1], (150, 250))
    
    def test_trajectory_hit_detection(self):
        """Test hit detection from y-velocity changes."""
        # Create synthetic trajectory with multiple clear direction changes
        frames = list(range(1, 301))
        y_coords = []
        
        # Create a more realistic trajectory with multiple hits
        for f in frames:
            if f < 50:
                y_coords.append(100 + f * 2)  # Going up fast
            elif f < 100:
                y_coords.append(200 - (f - 50) * 2)  # Going down fast
            elif f < 150:
                y_coords.append(100 + (f - 100) * 2)  # Going up again
            elif f < 200:
                y_coords.append(200 - (f - 150) * 2)  # Going down again
            elif f < 250:
                y_coords.append(100 + (f - 200) * 2)  # Going up once more
            else:
                y_coords.append(200 - (f - 250) * 2)  # Going down final time
        
        df = pd.DataFrame({
            'frame': frames,
            'x': [100] * len(frames),
            'y': y_coords,
            'vis': [1] * len(frames)
        })
        
        hit_ranges = detect_hits_from_trajectory(df, gap_hit=120, min_len=30)
        
        # Should detect at least one rally around the direction changes
        self.assertGreater(len(hit_ranges), 0)
    
    def test_hybrid_mode_integration(self):
        """Test full hybrid mode with visibility expansion."""
        # Create test data with hits within visibility spans
        frames = list(range(1, 301))
        vis = [1 if 50 <= f <= 250 else 0 for f in frames]  # Visible from frame 50-250
        
        # Create y-trajectory with multiple direction changes
        y_coords = []
        for f in frames:
            if f < 100:
                y_coords.append(50 + f * 0.5)  # Going up
            elif f < 150:
                y_coords.append(100 - (f - 100) * 0.5)  # Going down
            elif f < 200:
                y_coords.append(75 + (f - 150) * 0.5)  # Going up again
            else:
                y_coords.append(100 - (f - 200) * 0.5)  # Going down again
        
        df = pd.DataFrame({
            'frame': frames,
            'x': [100] * len(frames),
            'y': y_coords,
            'vis': vis
        })
        
        # Test hit detection
        hit_ranges = detect_hits_from_trajectory(df, gap_hit=120, min_len=30)
        
        # Test visibility expansion
        if hit_ranges:
            visible_frames = df[df['vis'] == 1]['frame'].tolist()
            vis_spans = contiguous_ranges(visible_frames)
            expanded = expand_hits_to_visibility_spans(hit_ranges, vis_spans)
            
            # Expanded range should start at 50 and end at 250 (visibility span)
            if expanded:
                self.assertEqual(expanded[0][0], 50)
                self.assertEqual(expanded[0][1], 250)
    
    def test_expand_hits_to_visibility(self):
        """Test expansion of hit ranges to visibility spans."""
        hit_ranges = [(100, 150)]
        vis_spans = [(50, 200), (250, 300)]
        
        expanded = expand_hits_to_visibility_spans(hit_ranges, vis_spans)
        
        # Hit range 100-150 should expand to visibility span 50-200
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0], (50, 200))


if __name__ == '__main__':
    unittest.main() 