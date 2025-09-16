# Models Directory

This directory is intended to store the AI models used by the person tracking system.

## What Goes Here

### 1. YOLOv8 Models
The person detection system uses YOLOv8 models for robust person detection. These models will be automatically downloaded when first used:

- **yolov8n.pt** - YOLOv8 Nano (fastest, default)
- **yolov8s.pt** - YOLOv8 Small (balanced speed/accuracy) 
- **yolov8m.pt** - YOLOv8 Medium (higher accuracy)
- **yolov8l.pt** - YOLOv8 Large (best accuracy, slower)
- **yolov8x.pt** - YOLOv8 XLarge (highest accuracy, slowest)

### 2. DeepSORT Models
DeepSORT tracking uses appearance embeddings for person re-identification:

- **deep_sort.pb** - DeepSORT feature extractor (automatically handled by deep-sort-realtime library)
- **mars-small128.pb** - Alternative appearance model
- **market1501.pb** - Market1501 trained model

## Automatic Model Download

**The models folder starts empty by design.** Models are automatically downloaded when the system runs for the first time:

1. **YOLOv8 Models**: Downloaded by Ultralytics when PersonDetector is first initialized
2. **DeepSORT Models**: Downloaded by deep-sort-realtime library when tracker is initialized

## Model Storage Locations

### YOLOv8 Models
By default, YOLOv8 models are stored in:
- **Windows**: `C:\Users\{username}\.ultralytics\`
- **Linux/Mac**: `~/.ultralytics/`

### DeepSORT Models  
DeepSORT models are stored by the library in:
- **Windows**: `C:\Users\{username}\.deep_sort_realtime\`
- **Linux/Mac**: `~/.deep_sort_realtime/`

## Manual Model Placement

If you want to use custom models or avoid automatic downloads:

### Custom YOLOv8 Model
```python
# Place your custom model in this folder, e.g.:
models/custom_yolo.pt

# Then update config.py:
YOLO_MODEL = "models/custom_yolo.pt"
```

### Pre-downloaded Models
You can manually download and place models here:

```bash
# Download YOLOv8 nano model
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -P models/

# Update config to use local model
YOLO_MODEL = "models/yolov8n.pt"
```

## Model Sizes and Performance

| Model | Size | Speed | mAP | Use Case |
|-------|------|-------|-----|----------|
| yolov8n.pt | 6.2MB | Fastest | 37.3 | Real-time, low resources |
| yolov8s.pt | 21.5MB | Fast | 44.9 | Balanced performance |
| yolov8m.pt | 49.7MB | Medium | 50.2 | Higher accuracy needed |
| yolov8l.pt | 83.7MB | Slow | 52.9 | Best accuracy |
| yolov8x.pt | 136.7MB | Slowest | 53.9 | Maximum accuracy |

## Configuration

The model selection is controlled in `src/config.py`:

```python
class Config:
    # Default model (will be auto-downloaded)
    YOLO_MODEL = "yolov8n.pt"  
    
    # To use a local model in this folder:
    # YOLO_MODEL = "models/yolov8n.pt"
```

## First Run Example

When you first run the tracker:

```bash
python src/main.py --video input.mp4
```

You'll see output like:
```
Downloading https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt...
100%|██████████| 6.23M/6.23M [00:01<00:00, 4.56MB/s]
Loaded YOLOv8 model: yolov8n.pt
```

## Troubleshooting

### Model Download Issues
If automatic download fails:
1. Check internet connection
2. Manually download models to this folder
3. Update config.py to use local paths

### Disk Space
- YOLOv8n (default): ~6MB
- DeepSORT models: ~30MB  
- Total: ~40MB for basic setup

### Custom Models
To use your own trained models:
1. Place .pt file in this folder
2. Update YOLO_MODEL in config.py
3. Ensure model is compatible with Ultralytics YOLOv8

## Model Licenses

- **YOLOv8**: AGPL-3.0 (Ultralytics)
- **DeepSORT**: MIT License (various implementations)

Check individual model licenses before commercial use.
