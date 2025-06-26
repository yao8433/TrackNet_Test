# TrackNetV3 Model Files

Place your TrackNetV3 model files in this directory:

## Required Files

1. **Model Weights**:
   ```
   models/tracknetv3/
   ├── tracknet_v3.pth          # PyTorch model weights
   ├── tracknet_v3_weights.h5   # Keras/TensorFlow weights (if applicable)
   └── model_config.json        # Model configuration
   ```

2. **Optional Files**:
   ```
   models/tracknetv3/
   ├── preprocessing_params.json  # Normalization parameters
   ├── class_mapping.json        # Label mappings if needed
   └── model_architecture.py     # Custom model definition (if needed)
   ```

## Download Instructions

If you have the TrackNetV3 model from a repository or paper:

1. Download the model weights file (usually `.pth` for PyTorch)
2. Place it in this directory: `models/tracknetv3/`
3. Update the model path in `src/tracknetv3_infer.py`

## Model Path Configuration

The default model path in the code will be:
```python
MODEL_PATH = "models/tracknetv3/tracknet_v3.pth"
```

Update `src/tracknetv3_infer.py` to load your specific model file. 