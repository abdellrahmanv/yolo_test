"""
YOLO to TFLite Exporter
Run this on your PC (not Raspberry Pi) to export models to TFLite INT8 format.
"""

from ultralytics import YOLO
import os

print("="*60)
print("🚀 YOLO to TFLite INT8 Exporter")
print("="*60)
print()

# Models to export
models_to_export = [
    ("models/yolov8n.pt", "YOLOv8n"),
    ("models/yolo11n.pt", "YOLO11n")
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
    print("1. Copy the .tflite files to: tensorlite/models/")
    print()
    print("   Example:")
    for export_path, model_name in exported_files:
        filename = os.path.basename(export_path)
        print(f"   cp {export_path} tensorlite/models/{filename}")
    print()
    print("2. Transfer to Raspberry Pi:")
    print("   scp tensorlite/models/*.tflite pi@raspberrypi:~/yolo_test/tensorlite/models/")
    print()
    print("3. On Raspberry Pi:")
    print("   cd yolo_test/tensorlite")
    print("   ./setup_tflite.sh")
    print("   ./run_yolo8n_tflite.sh")
    print("="*60)
else:
    print("❌ No models were exported successfully.")
    print()
    print("💡 Make sure you have:")
    print("   - Installed ultralytics: pip install ultralytics")
    print("   - Model files in models/ directory, or")
    print("   - Internet connection to download models")
