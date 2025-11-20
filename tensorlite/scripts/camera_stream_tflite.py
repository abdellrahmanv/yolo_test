import cv2
import numpy as np
import time
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

def get_camera_stream(width=320, height=240):
    """
    Get camera stream optimized for TFLite on Raspberry Pi.
    Uses MJPG format for better performance.
    """
    try:
        print("🎥 Initializing camera for TFLite...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Test camera
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera initialized: {width}x{height} (MJPG)")
            return cap
    except Exception as e:
        print(f"⚠️  V4L2 MJPG failed: {e}")
    
    # Fallback to basic method
    try:
        cap = cv2.VideoCapture(0)
        cap.set(3, width)
        cap.set(4, height)
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera initialized: {width}x{height} (Basic)")
            return cap
    except Exception as e:
        print(f"❌ Camera initialization failed: {e}")
    
    return None
