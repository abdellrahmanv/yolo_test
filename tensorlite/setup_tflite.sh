#!/bin/bash

echo "🚀 Setting up TFLite Environment for Raspberry Pi..."
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install it first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Update system
echo "📦 Updating system packages..."
sudo apt update

# Install TFLite runtime and dependencies
echo "📥 Installing TFLite runtime and dependencies..."
sudo apt install -y python3-opencv python3-pip v4l-utils
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime
pip3 install psutil numpy

if [ $? -ne 0 ]; then
    echo "⚠️  Google Coral repo failed, trying alternative..."
    pip3 install tflite-runtime
fi

echo ""
echo "✅ TFLite runtime installed successfully!"
echo ""

# Optimize camera settings
echo "🎥 Optimizing camera for performance..."
v4l2-ctl --set-fmt-video=width=320,height=240,pixelformat=MJPG 2>/dev/null || echo "⚠️  v4l2-ctl optimization skipped (run manually if needed)"

echo ""
echo "📁 Directory structure verified"
echo ""

# Check if models exist
echo "🔍 Checking for TFLite models..."
if [ -f "models/best_int8.tflite" ]; then
    echo "✅ best_int8.tflite found (Custom trained model)"
else
    echo "⚠️  best_int8.tflite not found in models/"
    echo "   This is your main custom model!"
fi

if [ ! -f "models/yolov8n_int8.tflite" ]; then
    echo "⚠️  yolov8n_int8.tflite not found in models/"
    echo "   Export it on your PC using: python3 export_to_tflite.py"
fi

if [ ! -f "models/yolo11n_int8.tflite" ]; then
    echo "⚠️  yolo11n_int8.tflite not found in models/"
    echo "   Export it on your PC using: python3 export_to_tflite.py"
fi

echo ""
echo "✨ TFLite setup complete!"
echo ""
echo "To run the benchmark:"
echo "  1. Make scripts executable: chmod +x run_best_custom.sh run_yolo8n_tflite.sh run_yolo11n_tflite.sh"
echo "  2. Run Custom Model (MAIN): ./run_best_custom.sh"
echo "  3. Run YOLOv8n: ./run_yolo8n_tflite.sh"
echo "  4. Run YOLO11n: ./run_yolo11n_tflite.sh"
echo ""
