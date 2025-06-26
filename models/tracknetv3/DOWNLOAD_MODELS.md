# TrackNetV3 Model Files

The neural network model files are required but too large for Git. Download them separately:

## Required Files

1. **TrackNet_best.pt** (130MB) - Main tracking model
2. **InpaintNet_best.pt** (6MB) - Inpainting model

## Download Instructions

Place the model files in this directory (`models/tracknetv3/`) for the system to work properly.

### Expected Structure
```
models/tracknetv3/
├── TrackNet_best.pt      # Main model (130MB)
├── InpaintNet_best.pt    # Inpainting model (6MB)
├── README.md             # Original model documentation
└── DOWNLOAD_MODELS.md    # This file
```

## Model Architecture
- **Parameters**: 87-parameter TrackNetV3 architecture
- **Input**: 9 frames × 3 RGB channels (256×144 resolution)
- **Output**: 8-channel heatmap with shuttlecock coordinates
- **Optimization**: M1 MPS acceleration compatible

The system will automatically load these models when running the pipeline. 