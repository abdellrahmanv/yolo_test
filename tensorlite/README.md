# 🚀 YOLO TFLite Benchmark for Raspberry Pi 4B

TensorFlow Lite INT8 quantized models for faster inference on Raspberry Pi.

**Expected Performance:**
- YOLOv8n INT8: **5-7 FPS** @ 320x320
- YOLO11n INT8: **4-6 FPS** @ 320x320

(Much faster than PyTorch .pt models which only get 2-3 FPS!)

---

## 📁 Folder Structure

```
tensorlite/
│
├── models/
│     ├── yolov8n_int8.tflite
│     └── yolo11n_int8.tflite
│
├── scripts/
│     ├── run_yolo8n_tflite.py
│     ├── run_yolo11n_tflite.py
│     ├── camera_stream_tflite.py
│     ├── system_monitor.py
│     └── generate_report.py
│
├── logs/
│
├── export_to_tflite.py       ← Run on PC to export models
├── setup_tflite.sh           ← Setup script for Raspberry Pi
├── run_yolo8n_tflite.sh      ← Run YOLOv8n benchmark
└── run_yolo11n_tflite.sh     ← Run YOLO11n benchmark
```

---

## ⚙️ Setup

### Step 1: Export Models (On Your PC)

```bash
cd tensorlite
python3 export_to_tflite.py
```

This will export both YOLOv8n and YOLO11n to INT8 TFLite format.

**Or manually:**
```python
from ultralytics import YOLO

# Export YOLOv8n
model = YOLO("yolov8n.pt")
model.export(format="tflite", int8=True)

# Export YOLO11n
model = YOLO("yolo11n.pt")
model.export(format="tflite", int8=True)
```

### Step 2: Copy Models to Raspberry Pi

Transfer the `.tflite` files to `tensorlite/models/` directory.

### Step 3: Setup on Raspberry Pi

```bash
cd tensorlite
chmod +x setup_tflite.sh run_yolo8n_tflite.sh run_yolo11n_tflite.sh
./setup_tflite.sh
```

This will:
- Install TFLite runtime
- Install OpenCV and dependencies
- Optimize camera settings for MJPG format

---

## 🚀 Run Benchmark

### YOLOv8n TFLite
```bash
./run_yolo8n_tflite.sh
```

### YOLO11n TFLite
```bash
./run_yolo11n_tflite.sh
```

### Manual Run
```bash
python3 scripts/run_yolo8n_tflite.py
python3 scripts/run_yolo11n_tflite.py
```

---

## 📊 Output

Each test generates:

1. **CSV File**: `logs/yolo8n_tflite.csv` or `logs/yolo11n_tflite.csv`
   - Columns: frame, fps, cpu, ram, temp

2. **Markdown Report**: `logs/yolo8n_tflite_report.md` or `logs/yolo11n_tflite_report.md`
   - Summary statistics
   - ASCII graphs for FPS, CPU, RAM, Temperature

---

## 🎯 Features

- ⚡ **2-3x faster** than PyTorch models
- 📊 **Auto-generated reports** with graphs
- 🛑 **Ctrl+C support** - generates report even when interrupted
- 🎥 **Live camera view** (if display available)
- 📈 **Real-time metrics** - FPS, CPU%, RAM%, Temperature
- 🔧 **Optimized** for Raspberry Pi 4B with camera MJPG format

---

## 🔧 Camera Optimization (Optional)

For best performance, optimize camera settings:

```bash
v4l2-ctl --set-fmt-video=width=320,height=240,pixelformat=MJPG
```

Disable power saving in `raspi-config`:
```bash
sudo raspi-config
# → Performance Options → Disable power saving
```

---

## 📝 Notes

- TFLite INT8 models are quantized for faster inference on ARM CPUs
- No GPU acceleration needed - runs on CPU efficiently
- Models are smaller in size (~6MB vs ~12MB for .pt)
- Ideal for real-time applications on edge devices

---

*TensorFlow Lite INT8 - Optimized for Raspberry Pi*
