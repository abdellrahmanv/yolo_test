# TFLite Runtime Installation Guide

## Why TFLite Runtime?

- **Lightweight**: Only 5 MB (vs 500+ MB for TensorFlow)
- **Fast**: Optimized for edge devices like Raspberry Pi
- **Efficient**: INT8 quantization for better performance

---

## Installation Methods

### Method 1: Google Coral Repository (Recommended)

```bash
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime
```

### Method 2: PyPI

```bash
pip3 install tflite-runtime
```

### Method 3: User Install (if pip fails)

```bash
pip3 install --user tflite-runtime
```

### Method 4: System Package Manager

```bash
sudo apt update
sudo apt install python3-tflite-runtime
```

---

## Verify Installation

```bash
python3 -c "import tflite_runtime.interpreter as tflite; print('✅ SUCCESS: tflite_runtime installed')"
```

---

## Common Issues

### Issue 1: "No matching distribution found"

**Solution**: Check your Python version and architecture:

```bash
python3 --version
uname -m
```

TFLite runtime supports:
- Python 3.7, 3.8, 3.9, 3.10, 3.11
- ARM architectures (armv7l, aarch64)

### Issue 2: "externally-managed-environment"

**Solution**: Use virtual environment or --break-system-packages:

```bash
# Option A: Virtual environment (recommended)
python3 -m venv ~/tflite_env
source ~/tflite_env/bin/activate
pip install tflite-runtime

# Option B: Break system packages (not recommended)
pip3 install tflite-runtime --break-system-packages
```

### Issue 3: Import fails after installation

**Solution**: Check installation location:

```bash
pip3 show tflite-runtime
python3 -c "import sys; print('\n'.join(sys.path))"
```

---

## Alternative: OpenCV DNN (No TFLite needed!)

If TFLite installation keeps failing, use OpenCV DNN instead:

```bash
# OpenCV is already installed, just run:
python3 scripts/run_best_custom_opencv.py
```

**Note**: OpenCV DNN might not support all TFLite model formats. If you see errors, you'll need to install tflite-runtime.

---

## Still Having Issues?

1. Check your Python version matches tflite-runtime compatibility
2. Try installing in a virtual environment
3. Check if pip is up to date: `pip3 install --upgrade pip`
4. Verify architecture: `uname -m` (should be armv7l or aarch64)
5. Check available wheels: `pip3 index versions tflite-runtime`
