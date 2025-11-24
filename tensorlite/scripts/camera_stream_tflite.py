import subprocess
import numpy as np
import threading
import queue
import time


class RpicamVideoStream:
    """Stream video using rpicam-vid command (no libcamera Python needed)"""
    
    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height
        self.frame_queue = queue.Queue(maxsize=2)
        self.stopped = False
        
        # Start rpicam-vid process streaming raw YUV420 to stdout
        self.process = subprocess.Popen(
            [
                'rpicam-vid',
                '-t', '0',  # Run indefinitely
                '--width', str(width),
                '--height', str(height),
                '--framerate', '30',
                '-o', '-',  # Output to stdout
                '--codec', 'yuv420',
                '-n'  # No preview window
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=width * height * 3 // 2
        )
        
        # Start thread to read frames
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        
        # Wait for first frame
        time.sleep(2)
        print(f"RpicamVideoStream initialized: {width}x{height}")
    
    def _reader(self):
        """Background thread to read frames"""
        frame_size = self.width * self.height * 3 // 2  # YUV420 size
        
        while not self.stopped:
            try:
                # Read one frame worth of YUV420 data
                raw_frame = self.process.stdout.read(frame_size)
                
                if len(raw_frame) != frame_size:
                    break
                
                # Convert YUV420 to RGB
                yuv = np.frombuffer(raw_frame, dtype=np.uint8)
                y = yuv[:self.width * self.height].reshape((self.height, self.width))
                u = yuv[self.width * self.height:self.width * self.height + self.width * self.height // 4].reshape((self.height // 2, self.width // 2))
                v = yuv[self.width * self.height + self.width * self.height // 4:].reshape((self.height // 2, self.width // 2))
                
                # Upsample U and V
                u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
                v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)
                
                # Convert to RGB
                r = y + 1.402 * (v - 128)
                g = y - 0.344136 * (u - 128) - 0.714136 * (v - 128)
                b = y + 1.772 * (u - 128)
                
                rgb = np.stack([r, g, b], axis=2)
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                
                # Update queue
                if not self.frame_queue.full():
                    self.frame_queue.put(rgb)
                
            except Exception as e:
                print(f"Frame read error: {e}")
                break
    
    def read(self):
        """Read a frame"""
        if self.stopped or self.frame_queue.empty():
            return False, None
        try:
            frame = self.frame_queue.get(timeout=1)
            return True, frame
        except:
            return False, None
    
    def release(self):
        """Release resources"""
        self.stopped = True
        if self.process:
            self.process.terminate()
            self.process.wait()
    
    def isOpened(self):
        """Check if stream is open"""
        return not self.stopped


def get_camera_stream(width=320, height=240):
    """
    Get camera stream using rpicam-vid command.
    Works with rpicam commands, no libcamera Python bindings needed.
    """
    print("Initializing camera with rpicam-vid...")
    
    # Check if rpicam-vid is available
    try:
        result = subprocess.run(['which', 'rpicam-vid'], capture_output=True)
        if result.returncode != 0:
            print("ERROR: rpicam-vid not found!")
            print("Make sure rpicam-apps is installed")
            return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    
    try:
        stream = RpicamVideoStream(width, height)
        print(f"Camera initialized: {width}x{height} (rpicam-vid)")
        return stream
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return None