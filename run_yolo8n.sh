#!/bin/bash

# Run YOLOv8n benchmark

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first"
    exit 1
fi

echo "🚀 Starting YOLOv8n Benchmark..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Run benchmark
python3 scripts/run_yolo8n.py

# Deactivate when done
deactivate

echo ""
echo "✅ YOLOv8n benchmark complete!"
echo "📊 Results: logs/yolo8n.csv"
echo "📈 Report: logs/yolo8n_report.md"
