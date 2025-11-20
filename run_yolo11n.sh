#!/bin/bash

# Run YOLO11n benchmark

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first"
    exit 1
fi

echo "🚀 Starting YOLO11n Benchmark..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Run benchmark
python3 scripts/run_yolo11n.py

# Deactivate when done
deactivate

echo ""
echo "✅ YOLO11n benchmark complete!"
echo "📊 Results: logs/yolo11n.csv"
echo "📈 Report: logs/yolo11n_report.md"
