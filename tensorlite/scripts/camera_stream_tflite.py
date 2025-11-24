import cv2
import time


def get_camera_stream(width=320, height=240):
    """
    Get camera stream using OpenCV for Raspberry Pi.
    Works with any camera accessible via /dev/video0
    """
    print("Initializing camera with OpenCV...")
    
    # Try different camera backends
    backends = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_ANY, "ANY"),
    ]
    
    for backend, name in backends:
        try:
            print(f"Trying {name} backend...")
            cap = cv2.VideoCapture(0, backend)
            
            if not cap.isOpened():
                cap.release()
                continue
            
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Try MJPEG format for better performance
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # Test read
            ret, frame = cap.read()
            if ret and frame is not None:
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Camera initialized: {actual_w}x{actual_h} ({name} backend)")
                return cap
            
            cap.release()
        except Exception as e:
            print(f"{name} backend failed: {e}")
            continue
    
    # If all backends fail, try basic approach
    try:
        print("Trying basic OpenCV VideoCapture...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        ret, frame = cap.read()
        if ret:
            print(f"Camera initialized: {width}x{height} (basic mode)")
            return cap
    except Exception as e:
        print(f"Basic OpenCV failed: {e}")
    
    print("\nERROR: Could not open camera!")
    print("Troubleshooting:")
    print("  1. Check camera connection: rpicam-hello")
    print("  2. Check video device: ls -l /dev/video*")
    print("  3. Add user to video group: sudo usermod -a -G video $USER")
    print("  4. Reboot if needed")
    
    return None