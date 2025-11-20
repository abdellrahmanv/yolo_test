"""
TFLite model exporter for YOLO models.
Run this on your PC (not Raspberry Pi) to export .pt models to .tflite format.
"""

from ultralytics import YOLO
import os

def export_models():
    """Export YOLOv8n and YOLO11n to TFLite INT8 format"""
    
    models = {
        "yolov8n.pt": "yolov8n_int8.tflite",
        "yolo11n.pt": "yolo11n_int8.tflite"
    }
    
    for pt_model, tflite_name in models.items():
        print(f"\n🔄 Exporting {pt_model} to TFLite INT8...")
        
        if not os.path.exists(f"../models/{pt_model}"):
            print(f"❌ Model not found: ../models/{pt_model}")
            print(f"💡 Download it first or run from yolo_test directory")
            continue
        
        try:
            model = YOLO(f"../models/{pt_model}")
            model.export(format="tflite", int8=True)
            
            print(f"✅ Exported successfully!")
            print(f"📦 Output: {tflite_name}")
            print(f"📋 Move it to: tensorlite/models/{tflite_name}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    print("\n" + "="*50)
    print("📝 NEXT STEPS:")
    print("1. Copy the generated .tflite files to tensorlite/models/")
    print("2. On Raspberry Pi, run: ./tensorlite/setup_tflite.sh")
    print("3. Run benchmark: ./tensorlite/run_yolo8n_tflite.sh")
    print("="*50)

if __name__ == "__main__":
    print("="*50)
    print("🚀 YOLO TFLite Exporter")
    print("="*50)
    export_models()
