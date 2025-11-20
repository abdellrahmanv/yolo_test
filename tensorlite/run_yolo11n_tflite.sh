#!/bin/bash

echo "🚀 Starting YOLO11n TFLite Benchmark..."
echo ""

# Run benchmark
python3 scripts/run_yolo11n_tflite.py

echo ""
echo "✅ YOLO11n TFLite benchmark complete!"
echo "📊 Results: logs/yolo11n_tflite.csv"
echo "📈 Report: logs/yolo11n_tflite_report.md"
