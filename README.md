# 🚀 YOLO Benchmark for Raspberry Pi 4B

Automated benchmarking architecture for comparing YOLOv8n and YOLOv11n performance on Raspberry Pi.

## 📁 Folder Structure

```
yolo_test/
│
├── models/
│     ├── yolov8n.pt
│     └── yolov11n.pt
│
├── scripts/
│     ├── run_benchmark.py     ← runs both models automatically
│     ├── system_monitor.py
│     └── camera_stream.py
│
└── logs/
      ├── yolo8n.csv
      ├── yolo11n.csv
      └── detailed/
```

## ⚙️ Setup on Raspberry Pi

### Automatic Setup (Recommended)

1. **Make setup script executable and run:**
   ```bash
   cd yolo_test
   chmod +x setup.sh run.sh
   ./setup.sh
   ```

   This will:
   - Create a virtual environment
   - Install all dependencies (ultralytics, opencv-python, psutil)
   - Set up directory structure

2. **Download models** (optional - script can auto-download):
   ```bash
   cd models
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
   ```

3. **Connect camera** to Raspberry Pi

### Manual Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Run Benchmark

### Quick Run (with auto-activation)

```bash
./run.sh
```

### Manual Run

```bash
source venv/bin/activate
python3 scripts/run_benchmark.py
deactivate
```

This will:
- Run YOLOv8n for 30 seconds → `logs/yolo8n.csv`
- Run YOLOv11n for 30 seconds → `logs/yolo11n.csv`

## 📊 What Gets Logged

Each CSV contains per-frame data:
- **FPS** - Frames per second
- **CPU** - CPU usage percentage
- **RAM** - RAM usage percentage  
- **Temp** - Raspberry Pi temperature (°C)

## 🔧 Configuration

Edit `run_benchmark.py` to adjust:
- `TEST_DURATION` - Test length in seconds (default: 30)
- `IMG_SIZE` - Input image size (default: 320)
- Camera resolution in `camera_stream.py` (default: 320x240)

## 📈 Fair Comparison

Both models are tested with identical:
- Camera settings
- Resolution
- Test duration
- Image size
- Logging format

This ensures 100% fair performance comparison.
