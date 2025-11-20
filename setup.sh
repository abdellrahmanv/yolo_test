#!/bin/bash

echo "🚀 Setting up YOLO Benchmark Environment for Raspberry Pi..."
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install it first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

if [ ! -d "venv" ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
echo "   - ultralytics (YOLO)"
echo "   - opencv-python"
echo "   - psutil"
echo "   - picamera2 (Raspberry Pi camera support)"
echo ""

# Install picamera2 system package first (needs to be from apt)
echo "📦 Installing picamera2 from system packages..."
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera

pip install ultralytics opencv-python psutil numpy

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "✅ All dependencies installed successfully!"
echo ""

# Create logs directory if not exists
mkdir -p logs/detailed

echo "📁 Directory structure verified"
echo ""

# Check if models exist
echo "🔍 Checking for YOLO models..."
if [ ! -f "models/yolov8n.pt" ]; then
    echo "⚠️  yolov8n.pt not found in models/"
    echo "   Download it with: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/"
fi

if [ ! -f "models/yolov11n.pt" ]; then
    echo "⚠️  yolov11n.pt not found in models/"
    echo "   Download it with: wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -P models/"
fi

echo ""
echo "✨ Setup complete!"
echo ""
echo "To run the benchmark:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the benchmark: python3 scripts/run_benchmark.py"
echo ""
echo "To deactivate the virtual environment later: deactivate"
echo ""
