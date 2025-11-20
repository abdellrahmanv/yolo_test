# YOLO TFLite Models (INT8 Quantized)

Place your TensorFlow Lite model files here:

- `yolov8n_int8.tflite` - YOLOv8 Nano INT8 quantized
- `yolo11n_int8.tflite` - YOLO11 Nano INT8 quantized

## How to Generate TFLite Models

### On Your PC (not Raspberry Pi):

1. **Install Ultralytics:**
   ```bash
   pip install ultralytics
   ```

2. **Export YOLOv8n to TFLite:**
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   model.export(format="tflite", int8=True)
   ```

3. **Export YOLO11n to TFLite:**
   ```python
   from ultralytics import YOLO
   model = YOLO("yolo11n.pt")
   model.export(format="tflite", int8=True)
   ```

4. **Copy the generated `.tflite` files to this directory**

### Or Use the Export Script:

From the `tensorlite/` directory:
```bash
python3 export_to_tflite.py
```

This will export both models automatically (if .pt files exist in ../models/).

## Expected Performance on Raspberry Pi 4B

- **YOLOv8n INT8**: 5-7 FPS @ 320x320
- **YOLO11n INT8**: 4-6 FPS @ 320x320

Much faster than PyTorch (.pt) which only gets 2-3 FPS!
