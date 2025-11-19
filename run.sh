#!/bin/bash

# Quick run script - automatically activates venv and runs benchmark

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first"
    exit 1
fi

echo "🚀 Starting YOLO Benchmark..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Run benchmark
python3 scripts/run_benchmark.py

# Deactivate when done
deactivate

echo ""
echo "✅ Benchmark complete! Check logs/ directory for results."
