import cv2
import subprocess
import os
import numpy as np
from picamera2 import Picamera2
from threading import Thread
import time

class RpiCameraStream:
    """Camera stream using picamera2 for Raspberry Pi"""
    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height
        self.camera = Picamera2()
        
        # Configure camera
        config = self.camera.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.camera.configure(config)
        self.camera.start()
        
        # Warm up camera
        time.sleep(2)
        self.frame = None
        self.stopped = False
        
    def read(self):
        """Read frame from camera"""
        if self.stopped:
            return False, None
        frame = self.camera.capture_array()
        return True, frame
    
    def release(self):
        """Release camera"""
        self.stopped = True
        self.camera.stop()
        self.camera.close()
    
    def isOpened(self):
        """Check if camera is opened"""
        return not self.stopped
    
    def set(self, prop, value):
        """Dummy set method for compatibility"""
        pass

def get_camera_stream(width=320, height=240):
    """
    Get camera stream for Raspberry Pi using picamera2 (libcamera).
    """
    try:
        print("🎥 Initializing camera with picamera2 (libcamera)...")
        cam = RpiCameraStream(width, height)
        print(f"✅ Camera initialized: {width}x{height} (picamera2)")
        return cam
    except Exception as e:
        print(f"❌ picamera2 failed: {e}")
        print("💡 Install picamera2: sudo apt install -y python3-picamera2")
        return None
