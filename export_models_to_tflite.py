"""
YOLO to TFLite Exporter
⚠️  IMPORTANT: Export should be done on a PC, not Raspberry Pi!
TensorFlow is too heavy for Raspberry Pi.
"""

from ultralytics import YOLO
import os
import sys

print("="*60)
print("🚀 YOLO to TFLite INT8 Exporter")
print("="*60)
print()

# Detect where we are running
if os.path.exists("/home/pi"):
    print("⚠️  WARNING: You are running this on Raspberry Pi!")
    print("   TensorFlow export is very slow and may fail on Pi.")
    print()
    print("💡 RECOMMENDED: Export on your PC instead:")
    print("   1. Clone this repo on your PC")
    print("   2. Run: python export_models_to_tflite.py")
    print("   3. Copy the .tflite files to Pi")
    print()
    response = input("Continue anyway? (y/N): ")
    if response.lower() != 'y':
        print("Exiting...")
        sys.exit(0)
    base_path = "/home/pi/yolo_test"
    print("📍 Running on Raspberry Pi")
else:
    # Running on PC
    base_path = os.path.dirname(os.path.abspath(__file__))
    print("📍 Detected: Running on PC (recommended)")

print(f"📁 Working directory: {base_path}")
print()

# Models to export
models_to_export = [
    (os.path.join(base_path, "models/yolov8n.pt"), "YOLOv8n"),
    (os.path.join(base_path, "models/yolo11n.pt"), "YOLO11n")
]

exported_files = []

for model_path, model_name in models_to_export:
    print(f"📦 Exporting {model_name}...")
    print(f"   Source: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"   ⚠️  File not found! Downloading from Ultralytics...")
        try:
            # Extract just the filename for download
            model_file = os.path.basename(model_path)
            model = YOLO(model_file)
            print(f"   ✅ Downloaded: {model_file}")
        except Exception as e:
            print(f"   ❌ Failed to download: {e}")
            continue
    else:
        model = YOLO(model_path)
        print(f"   ✅ Loaded from disk")
    
    try:
        # Export to TFLite INT8
        print(f"   🔄 Exporting to TFLite INT8 format...")
        export_path = model.export(format="tflite", int8=True)
        print(f"   ✅ Export successful!")
        print(f"   📁 Output: {export_path}")
        exported_files.append((export_path, model_name))
        print()
    except Exception as e:
        print(f"   ❌ Export failed: {e}")
        print()

# Summary
print("="*60)
print("📊 Export Summary")
print("="*60)

if exported_files:
    print(f"✅ Successfully exported {len(exported_files)} model(s):\n")
    for export_path, model_name in exported_files:
        print(f"   • {model_name}: {export_path}")
    
    print("\n" + "="*60)
    print("📝 NEXT STEPS:")
    print("="*60)
    
    # Create tensorlite/models directory if it doesn't exist
    tflite_models_dir = os.path.join(base_path, "tensorlite/models")
    os.makedirs(tflite_models_dir, exist_ok=True)
    
    # Copy files to tensorlite/models/
    print("📦 Copying files to tensorlite/models/...")
    for export_path, model_name in exported_files:
        filename = os.path.basename(export_path)
        dest_path = os.path.join(tflite_models_dir, filename)
        try:
            import shutil
            shutil.copy2(export_path, dest_path)
            print(f"   ✅ Copied: {filename}")
        except Exception as e:
            print(f"   ⚠️  Manual copy needed: {export_path} → {dest_path}")
    
    print()
    if os.path.exists("/home/pi"):
        print("🎯 You're on Raspberry Pi! Run the benchmark:")
        print("   cd tensorlite")
        print("   ./setup_tflite.sh")
        print("   ./run_yolo8n_tflite.sh")
    else:
        print("🎯 Transfer to Raspberry Pi:")
        print("   scp -r tensorlite pi@raspberrypi:~/yolo_test/")
    print("="*60)
else:
    print("❌ No models were exported successfully.")
    print()
    print("💡 Make sure you have:")
    print("   - Installed ultralytics: pip install ultralytics")
    print("   - Model files in models/ directory, or")
    print("   - Internet connection to download models")
