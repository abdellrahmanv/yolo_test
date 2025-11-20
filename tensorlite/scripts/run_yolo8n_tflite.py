import cv2
import numpy as np
import time, csv, os, sys, signal
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

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

MODEL_NAME = "yolo8n"
MODEL_PATH = os.path.join(project_root, "models/yolov8n_int8.tflite")

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
    
    print(f"\n=== Running TFLite test for {MODEL_NAME} ===\n")
    print(f"💡 Press Ctrl+C or 'q' to stop and generate report\n")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print(f"💡 Run export_to_tflite.py on your PC first!")
        return
    
    # Load TFLite interpreter
    print("📦 Loading TFLite model...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"✅ Model loaded: {MODEL_PATH}")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Input dtype: {input_details[0]['dtype']}")
    
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
            
            # Preprocess for TFLite
            img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.uint8)
            img = np.expand_dims(img, axis=0)
            
            # Run inference
            t0 = time.time()
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            outputs = interpreter.get_tensor(output_details[0]['index'])
            inference_time = time.time() - t0
            fps = 1 / inference_time
            
            cpu, ram, temp = get_system_stats()
            writer.writerow([frame_count, fps, cpu, ram, temp])

            print(f"{MODEL_NAME} TFLite | Frame: {frame_count} | FPS: {fps:.2f} | CPU: {cpu}% | RAM: {ram}% | Temp: {temp}°C")
            
            # Display live view if display is available
            if SHOW_DISPLAY:
                try:
                    cv2.imshow(f'{MODEL_NAME} - TFLite INT8', frame)
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
