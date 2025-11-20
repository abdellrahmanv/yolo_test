import cv2
import subprocess
import os

def get_camera_stream(width=320, height=240):
    """
    Get camera stream for Raspberry Pi.
    Tries multiple methods: libcamera, picamera2, and standard OpenCV.
    """
    # Method 1: Try libcamera with OpenCV (Raspberry Pi OS Bullseye+)
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Test if camera works
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera initialized: {width}x{height} (V4L2)")
            return cap
    except Exception as e:
        print(f"⚠️  V4L2 method failed: {e}")
    
    # Method 2: Try standard OpenCV
    try:
        cap = cv2.VideoCapture(0)
        cap.set(3, width)
        cap.set(4, height)
        
        # Test if camera works
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera initialized: {width}x{height} (OpenCV default)")
            return cap
    except Exception as e:
        print(f"⚠️  OpenCV default method failed: {e}")
    
    # Method 3: Try legacy camera index
    try:
        cap = cv2.VideoCapture('/dev/video0')
        cap.set(3, width)
        cap.set(4, height)
        
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera initialized: {width}x{height} (/dev/video0)")
            return cap
    except Exception as e:
        print(f"⚠️  /dev/video0 method failed: {e}")
    
    print("❌ Failed to initialize camera with all methods")
    print("💡 Make sure camera is enabled: sudo raspi-config → Interface Options → Camera")
    return None
