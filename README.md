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

### Quick Run - Both Models (Sequential)

```bash
./run.sh
```

### Run Individual Models

**YOLOv8n only:**
```bash
chmod +x run_yolo8n.sh
./run_yolo8n.sh
```

**YOLO11n only:**
```bash
chmod +x run_yolo11n.sh
./run_yolo11n.sh
```

### Manual Run

```bash
source venv/bin/activate
python3 scripts/run_yolo8n.py   # For YOLOv8n
python3 scripts/run_yolo11n.py  # For YOLO11n
deactivate
```

This will:
- Run YOLOv8n for 30 seconds → `logs/yolo8n.csv`
- Run YOLOv11n for 30 seconds → `logs/yolo11n.csv`

## 📊 What Gets Logged

### CSV Files (Raw Data)
Each test generates a CSV in `logs/`:
- `logs/yolo8n.csv` - YOLOv8n raw data
- `logs/yolo11n.csv` - YOLO11n raw data

**Columns:** frame, fps, cpu, ram, temp, detections

### Markdown Reports (Visual Graphs)
Each test auto-generates a markdown report with ASCII graphs:
- `logs/yolo8n_report.md` - YOLOv8n visual report
- `logs/yolo11n_report.md` - YOLO11n visual report

**Includes:**
- 📊 Summary statistics table
- 📈 FPS graph over time
- 🔥 CPU usage graph
- 💾 RAM usage graph
- 🌡️ Temperature graph
- 🎯 Detections per frame graph

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
