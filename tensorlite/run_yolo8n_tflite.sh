#!/bin/bash

echo "🚀 Starting YOLOv8n TFLite Benchmark..."
echo ""

# Run benchmark
python3 scripts/run_yolo8n_tflite.py

echo ""
echo "✅ YOLOv8n TFLite benchmark complete!"
echo "📊 Results: logs/yolo8n_tflite.csv"
echo "📈 Report: logs/yolo8n_tflite_report.md"
