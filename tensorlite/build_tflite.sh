#!/bin/bash

echo "Building tflite-runtime for Python 3.13 on Raspberry Pi"
echo ""

# Install dependencies
echo "Installing build dependencies..."
sudo apt update
sudo apt install -y git cmake python3-pip python3-dev python3-numpy

# Clone TensorFlow
echo "Cloning TensorFlow repository..."
cd /tmp
if [ -d "tensorflow" ]; then
    rm -rf tensorflow
fi
git clone --depth 1 --branch v2.14.0 https://github.com/tensorflow/tensorflow.git
cd tensorflow

# Build tflite-runtime wheel for Python 3.13
echo "Building tflite-runtime wheel..."
PYTHON=python3 tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh native

# Install the built wheel
echo "Installing tflite-runtime..."
pip3 install /tmp/tensorflow/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/dist/*.whl

echo ""
echo "Done! tflite-runtime should now be installed for Python 3.13"
echo "Test it with: python3 -c \"import tflite_runtime.interpreter as tflite; print('Success!')\"
