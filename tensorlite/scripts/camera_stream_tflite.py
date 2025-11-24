import time

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    import cv2


class CameraWrapper:
    def __init__(self, camera_obj, is_picamera=False):
        self.camera = camera_obj
        self.is_picamera = is_picamera
        self.opened = True
    
    def read(self):
        if self.is_picamera:
            try:
                frame = self.camera.capture_array()
                return True, frame
            except:
                return False, None
        else:
            return self.camera.read()
    
    def release(self):
        if self.is_picamera:
            try:
                self.camera.stop()
            except:
                pass
        else:
            self.camera.release()
        self.opened = False
    
    def isOpened(self):
        return self.opened


def get_camera_stream(width=320, height=240):
    """
    Get camera stream optimized for TFLite on Raspberry Pi.
    Uses picamera2 for Raspberry Pi Camera Module.
    """
    if PICAMERA_AVAILABLE:
        try:
            print("Initializing Raspberry Pi Camera...")
            picam = Picamera2()
            config = picam.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            picam.configure(config)
            picam.start()
            time.sleep(2)
            print(f"Camera initialized: {width}x{height}")
            return CameraWrapper(picam, is_picamera=True)
        except Exception as e:
            print(f"Failed to initialize picamera2: {e}")
            return None
    else:
        print("ERROR: picamera2 not installed!")
        print("Install it: sudo apt install python3-picamera2")
        return None