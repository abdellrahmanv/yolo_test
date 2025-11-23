import cv2
import numpy as np
import time, csv, os, sys, signal

# Use Ultralytics YOLO to load TFLite models (works with Python 3.13!)
from ultralytics import YOLO

from system_monitor import get_system_stats
from camera_stream_tflite import get_camera_stream
from generate_report import generate_markdown_report

# Change to project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

TEST_DURATION = 30  # seconds
IMG_SIZE = 320
SHOW_DISPLAY = True

MODEL_NAME = "best_custom"
MODEL_PATH = os.path.join(project_root, "models/best_int8.tflite")

# Global flag for graceful shutdown
interrupted = False
log_path = None

def signal_handler(sig, frame):
    global interrupted
    print("\n\n⏹️  Test interrupted by user (Ctrl+C)")
    interrupted = True

def run_test():
    global SHOW_DISPLAY, interrupted, log_path
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"\n=== Running TFLite test for {MODEL_NAME} (Custom Model) ===\n")
    print(f"💡 Press Ctrl+C or 'q' to stop and generate report\n")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    # Load TFLite model using Ultralytics YOLO
    print("📦 Loading TFLite model with Ultralytics YOLO...")
    print(f"   Model path: {MODEL_PATH}")
    
    try:
        model = YOLO(MODEL_PATH, task='detect')
        print(f"✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    cap = get_camera_stream(width=IMG_SIZE, height=IMG_SIZE)
    
    # Check if camera opened successfully
    if cap is None or not cap.isOpened():
        print("❌ Error: Could not open camera!")
        return

    log_path = f"logs/{MODEL_NAME}_tflite.csv"
    start_time = time.time()
    frame_count = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "fps", "cpu", "ram", "temp"])

        while time.time() - start_time < TEST_DURATION and not interrupted:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame from camera")
                break
            
            frame_count += 1
            
            # Run inference with Ultralytics YOLO
            t0 = time.time()
            results = model(frame, imgsz=IMG_SIZE, verbose=False)
            inference_time = time.time() - t0
            fps = 1 / inference_time
            
            # Count detections
            detections = len(results[0].boxes) if results[0].boxes is not None else 0
            
            cpu, ram, temp = get_system_stats()
            writer.writerow([frame_count, fps, cpu, ram, temp])

            print(f"{MODEL_NAME} TFLite | Frame: {frame_count} | FPS: {fps:.2f} | Detections: {detections} | CPU: {cpu}% | RAM: {ram}% | Temp: {temp}°C")
            
            # Draw detections on frame
            annotated_frame = results[0].plot() if detections > 0 else frame
            
            # Display live view if display is available
            if SHOW_DISPLAY:
                try:
                    cv2.imshow(f'{MODEL_NAME} - Custom TFLite Model', annotated_frame)
                    # Press 'q' to quit early
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⏹️  Stopped by user (pressed 'q')")
                        interrupted = True
                        break
                except:
                    # If display fails, disable it
                    SHOW_DISPLAY = False
                    print("⚠️  Display not available, running in headless mode")

    cap.release()
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()
    
    # Always generate report, even if interrupted
    if frame_count > 0:
        print(f"\n✅ Test complete for {MODEL_NAME} TFLite! Collected {frame_count} frames")
        print(f"📊 Results saved to {log_path}")
        
        # Generate markdown report with graphs
        print(f"📈 Generating markdown report...")
        try:
            generate_markdown_report(MODEL_NAME, log_path)
        except Exception as e:
            print(f"⚠️  Failed to generate report: {e}")
    else:
        print(f"\n⚠️  No frames collected, skipping report generation")


if __name__ == "__main__":
    try:
        run_test()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted!")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
