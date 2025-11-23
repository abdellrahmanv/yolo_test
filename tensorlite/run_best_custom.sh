#!/bin/bash

echo "🚀 Starting Custom Model (best_int8) TFLite Benchmark..."
echo ""

# Run benchmark
python3 scripts/run_best_custom.py

echo ""
echo "✅ Custom model TFLite benchmark complete!"
echo "📊 Results: logs/best_custom_tflite.csv"
echo "📈 Report: logs/best_custom_tflite_report.md"
