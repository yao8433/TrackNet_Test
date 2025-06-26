#!/usr/bin/env python3
"""
Script to inspect TrackNetV3 model structure and understand the checkpoint format
"""

import torch
import os

def inspect_model(model_path):
    """Inspect the structure of a PyTorch model file."""
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    print(f"Inspecting model: {model_path}")
    print("=" * 60)
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print("Checkpoint keys:")
        for key in checkpoint.keys():
            print(f"  - {key}: {type(checkpoint[key])}")
        
        print("\n" + "=" * 60)
        
        # Check if it has a 'model' key
        if 'model' in checkpoint:
            model_state = checkpoint['model']
            print(f"Model state dict type: {type(model_state)}")
            
            if hasattr(model_state, 'state_dict'):
                print("Model has state_dict() method")
                state_dict = model_state.state_dict()
            elif isinstance(model_state, dict):
                print("Model state is a dictionary")
                state_dict = model_state
            else:
                print("Unknown model state format")
                return
            
            print(f"\nModel layers ({len(state_dict)} parameters):")
            for i, (name, param) in enumerate(state_dict.items()):
                if i < 20:  # Show first 20 layers
                    print(f"  {name}: {param.shape}")
                elif i == 20:
                    print(f"  ... ({len(state_dict) - 20} more layers)")
                    break
        
        # Check for direct state dict
        elif isinstance(checkpoint, dict) and any('conv' in k or 'fc' in k for k in checkpoint.keys()):
            print("Direct state dict found")
            print(f"\nModel layers ({len(checkpoint)} parameters):")
            for i, (name, param) in enumerate(checkpoint.items()):
                if i < 20:
                    print(f"  {name}: {param.shape}")
                elif i == 20:
                    print(f"  ... ({len(checkpoint) - 20} more layers)")
                    break
        
        # Additional info
        if 'epoch' in checkpoint:
            print(f"\nEpoch: {checkpoint['epoch']}")
        
        if 'param_dict' in checkpoint:
            print(f"Param dict: {checkpoint['param_dict']}")
        
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    model_paths = [
        "models/tracknetv3/TrackNet_best.pt",
        "models/tracknetv3/InpaintNet_best.pt"
    ]
    
    for path in model_paths:
        inspect_model(path)
        print("\n" + "=" * 80 + "\n") 