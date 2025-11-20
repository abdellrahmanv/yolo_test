# Pre-exported TFLite Models

## ⚠️ TensorFlow Export Issue on Raspberry Pi

Exporting YOLO models to TFLite requires TensorFlow, which is **too heavy for Raspberry Pi** and often fails.

## ✅ Solution: Use Pre-Exported Models

### Option 1: Download Pre-Exported Models (Recommended)

You can download pre-exported TFLite INT8 models:

**YOLOv8n INT8:**
```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n_saved_model.zip
unzip yolov8n_saved_model.zip
# Extract the .tflite file from the saved_model folder
```

**Or export on your PC:**

### Option 2: Export on Your Windows PC

1. **On your PC, install requirements:**
   ```powershell
   pip install ultralytics tensorflow
   ```

2. **Run the export script:**
   ```powershell
   cd C:\Users\Public\yolo_pi_benchmark\yolo_test
   python export_models_to_tflite.py
   ```

3. **Copy generated files to Pi:**
   ```powershell
   # The script will create .tflite files
   # Copy them using WinSCP, or:
   scp yolov8n_int8.tflite pi@<pi-ip>:~/yolo_test/tensorlite/models/
   scp yolo11n_int8.tflite pi@<pi-ip>:~/yolo_test/tensorlite/models/
   ```

### Option 3: Manual Export (Python Script)

On your PC, create a Python script:

```python
from ultralytics import YOLO

# Export YOLOv8n
model = YOLO("yolov8n.pt")
model.export(format="tflite", int8=True)

# Export YOLO11n  
model = YOLO("yolo11n.pt")
model.export(format="tflite", int8=True)
```

This will create:
- `yolov8n_int8.tflite`
- `yolo11n_int8.tflite`

### Once You Have the .tflite Files:

```bash
# On Raspberry Pi
cd ~/yolo_test/tensorlite/models
# Place the .tflite files here

cd ..
./setup_tflite.sh
./run_yolo8n_tflite.sh
```

---

## Why Not Export on Pi?

- TensorFlow installation is 500+ MB
- Export process is very slow (10+ minutes per model)
- Often fails due to memory constraints
- TFLite runtime (for inference) is only 5 MB and works perfectly!

**Bottom line:** Export on PC, run inference on Pi! 🚀
