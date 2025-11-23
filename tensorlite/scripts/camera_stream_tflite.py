import cv2
import numpy as np
import time

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False


def get_camera_stream(width=320, height=240):
    """
    Get camera stream optimized for TFLite on Raspberry Pi.
    Tries picamera2 first, then falls back to OpenCV.
    """
    # Try picamera2 first (native Raspberry Pi camera)
    if PICAMERA_AVAILABLE:
        try:
            print("Initializing Raspberry Pi Camera with picamera2...")
            picam = Picamera2()
            config = picam.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            picam.configure(config)
            picam.start()
            time.sleep(2)  # Camera warm-up
            print(f"Camera initialized: {width}x{height} (picamera2)")
            
            # Wrap picamera2 in a class that mimics cv2.VideoCapture
            class PiCameraWrapper:
                def __init__(self, picam):
                    self.picam = picam
                    self.opened = True
                
                def read(self):
                    try:
                        frame = self.picam.capture_array()
                        return True, frame
                    except:
                        return False, None
                
                def release(self):
                    try:
                        self.picam.stop()
                    except:
                        pass
                    self.opened = False
                
                def isOpened(self):
                    return self.opened
            
            return PiCameraWrapper(picam)
        except Exception as e:
            print(f"picamera2 failed: {e}")
    
    # Try OpenCV with V4L2 and MJPG
    try:
        print("Trying OpenCV with V4L2 MJPG...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        ret, frame = cap.read()
        if ret:
            print(f"Camera initialized: {width}x{height} (V4L2 MJPG)")
            return cap
        cap.release()
    except Exception as e:
        print(f"V4L2 MJPG failed: {e}")
    
    # Try basic OpenCV VideoCapture
    try:
        print("Trying basic OpenCV VideoCapture...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        ret, frame = cap.read()
        if ret:
            print(f"Camera initialized: {width}x{height} (OpenCV basic)")
            return cap
        cap.release()
    except Exception as e:
        print(f"Basic OpenCV failed: {e}")
    
    print("ERROR: Could not initialize any camera!")
    print("Please check:")
    print("  1. Camera is connected properly")
    print("  2. Camera is enabled: sudo raspi-config -> Interface Options -> Camera")
    print("  3. Run: libcamera-hello (to test camera)")
    print("  4. Check camera permissions: ls -l /dev/video*")
    
    return None