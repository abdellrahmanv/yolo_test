# TFLite Model Export Guide

## ⚠️ TFLite Export is Complex and Often Fails

TFLite export requires many dependencies that often conflict. Here are your best options:

## ✅ RECOMMENDED: Use ONNX Format Instead

ONNX is much easier to export and works well on Raspberry Pi with ONNX Runtime:

**On your PC:**
```python
from ultralytics import YOLO

# Export to ONNX (much simpler!)
model = YOLO("yolov8n.pt")
model.export(format="onnx")

model = YOLO("yolo11n.pt")
model.export(format="onnx")
```

This creates `yolov8n.onnx` and `yolo11n.onnx` - much more reliable!

## Alternative: Download Pre-Converted Models

### Option 1: Use Pre-exported Models from Ultralytics

Ultralytics provides pre-exported models:

```bash
# Download YOLOv8n TFLite
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.tflite

# Download YOLO11n TFLite  
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.tflite
```

### Option 2: Skip TFLite, Use PyTorch (.pt) Models

The PyTorch benchmarks already work on your Pi! Just use:
```bash
./run_yolo8n.sh
./run_yolo11n.sh
```

## Why TFLite Export Fails

Common issues:
- Missing `ai-edge-litert` package (not available on Windows)
- Complex `onnx_graphsurgeon` dependencies
- Tensorflow version conflicts
- Memory issues during conversion

## Best Practice for Raspberry Pi

**Use the PyTorch (.pt) benchmark** you already have working!
- It's simpler
- Already implemented
- Works reliably
- You get 2-3 FPS which is acceptable for benchmarking

The TFLite conversion was meant to give 5-7 FPS, but if it's causing problems, stick with what works! 🚀

---

*If you really need TFLite models, try exporting on a Linux machine with Docker, or use pre-converted models from Ultralytics Hub.*

