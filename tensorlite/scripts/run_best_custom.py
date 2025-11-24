import cv2
import numpy as np
import time, csv, os, sys, signal

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False

from system_monitor import get_system_stats
from camera_stream_tflite import get_camera_stream
from generate_report import generate_markdown_report

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

TEST_DURATION = 30
IMG_SIZE = 320
SHOW_DISPLAY = False

MODEL_NAME = "best_custom"
MODEL_PATH = os.path.join(project_root, "models/best_int8.tflite")

interrupted = False
log_path = None

def signal_handler(sig, frame):
    global interrupted
    print("\n\nTest interrupted by user (Ctrl+C)")
    interrupted = True

def run_test():
    global SHOW_DISPLAY, interrupted, log_path
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"\n=== Running TFLite test for {MODEL_NAME} (Custom Model) ===\n")
    print(f"Press Ctrl+C or 'q' to stop and generate report\n")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return
    
    if not TFLITE_AVAILABLE:
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(f"ERROR: tflite-runtime is not available for Python {py_version}")
        print("\nSOLUTION: Use Python 3.11 (tflite-runtime supports Python 3.7-3.11 only)")
        print("\nSteps to fix:")
        print("  1. Install Python 3.11:")
        print("     sudo apt install python3.11 python3.11-venv")
        print("  2. Create new venv with Python 3.11:")
        print("     python3.11 -m venv venv311")
        print("  3. Activate and install packages:")
        print("     source venv311/bin/activate")
        print("     pip install ultralytics tflite-runtime picamera2")
        print("  4. Run again with Python 3.11")
        return
    
    print(f"Loading TFLite model...")
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"Model loaded: {MODEL_PATH}")
        print(f"Input shape: {input_details[0]['shape']}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    cap = get_camera_stream(width=IMG_SIZE, height=IMG_SIZE)
    
    if cap is None or not cap.isOpened():
        print("Could not open camera!")
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
                print("Failed to grab frame")
                break
            
            frame_count += 1
            t0 = time.time()
            
            img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            outputs = interpreter.get_tensor(output_details[0]['index'])
            inference_time = time.time() - t0
            fps = 1 / inference_time
            
            cpu, ram, temp = get_system_stats()
            writer.writerow([frame_count, fps, cpu, ram, temp])

            print(f"{MODEL_NAME} TFLite | Frame: {frame_count} | FPS: {fps:.2f} | CPU: {cpu}% | RAM: {ram}% | Temp: {temp}C")
            
            if SHOW_DISPLAY:
                try:
                    cv2.imshow(f'{MODEL_NAME} - Custom TFLite Model', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopped by user (pressed 'q')")
                        interrupted = True
                        break
                except:
                    SHOW_DISPLAY = False
                    print("Display not available, running headless")

    cap.release()
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()
    
    if frame_count > 0:
        print(f"\nTest complete for {MODEL_NAME} TFLite! Collected {frame_count} frames")
        print(f"Results saved to {log_path}")
        
        print(f"Generating markdown report...")
        try:
            generate_markdown_report(MODEL_NAME, log_path)
        except Exception as e:
            print(f"Failed to generate report: {e}")
    else:
        print(f"\nNo frames collected, skipping report generation")


if __name__ == "__main__":
    try:
        run_test()
    except KeyboardInterrupt:
        print("\n\nTest interrupted!")
    except Exception as e:
        print(f"\nERROR during test: {e}")
        import traceback
        traceback.print_exc()


