import os
import sys

# Check if --no-display flag is passed to disable display
SHOW_DISPLAY = '--no-display' not in sys.argv

import cv2
import numpy as np
import time, csv, signal

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

MODEL_NAME = "best_custom"
MODEL_PATH = os.path.join(project_root, "models/best_int8.tflite")

interrupted = False
log_path = None

def signal_handler(sig, frame):
    global interrupted
    print("\n\nTest interrupted by user (Ctrl+C)")
    interrupted = True

def compute_ious(box1, boxes2):
    """Compute IoU between one box and an array of boxes."""
    x1 = np.maximum(box1[0], boxes2[:, 0])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y2 = np.minimum(box1[3], boxes2[:, 3])

    inter_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = box1_area + boxes2_area - inter_area
    return inter_area / (union_area + 1e-6)

def nms(boxes, scores, iou_threshold=0.4):
    """Non-Maximum Suppression to filter overlapping boxes."""
    if len(boxes) == 0:
        return [], []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    idxs = np.argsort(scores)[::-1]
    keep = []
    
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = compute_ious(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    
    return boxes[keep].tolist(), scores[keep].tolist()

def yolo_postprocess(output, conf_thresh=0.25, img_size=320):
    """
    Post-process YOLOv8 TFLite output to extract bounding boxes.
    Output shape is typically (1, 5, N) where N is number of anchor boxes.
    Each detection: [x_center, y_center, width, height, confidence]
    """
    boxes = []
    scores = []
    
    # Handle different output shapes
    if len(output.shape) == 3:
        detections = output[0]  # (5, N) or (N, 5)
        if detections.shape[0] == 5:
            detections = detections.T  # Convert to (N, 5)
    else:
        detections = output
    
    for detection in detections:
        if len(detection) >= 5:
            x, y, w, h, conf = detection[:5]
            if conf > conf_thresh:
                # Convert from normalized to pixel coordinates
                x_min = int((x - w/2) * img_size)
                y_min = int((y - h/2) * img_size)
                x_max = int((x + w/2) * img_size)
                y_max = int((y + h/2) * img_size)
                
                # Clip to image boundaries
                x_min = max(0, min(x_min, img_size))
                y_min = max(0, min(y_min, img_size))
                x_max = max(0, min(x_max, img_size))
                y_max = max(0, min(y_max, img_size))
                
                boxes.append([x_min, y_min, x_max, y_max])
                scores.append(float(conf))
    
    return boxes, scores

def run_test():
    global SHOW_DISPLAY, interrupted, log_path
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"\n=== Running TFLite test for {MODEL_NAME} (Custom Model) ===\n")
    if SHOW_DISPLAY:
        print(f"Display enabled - Press 'q' in window or Ctrl+C to stop\n")
        print(f"Tip: Use '--no-display' flag for headless operation\n")
    else:
        print(f"Running headless - Press Ctrl+C to stop and generate report\n")
    
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
        # Use 4 threads for better performance on Raspberry Pi 4
        interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
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
            
            # Convert BGR to RGB FIRST (OpenCV uses BGR, model expects RGB)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
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
                    # Display the correctly processed RGB image (converted back to BGR for OpenCV)
                    img_disp = cv2.cvtColor((img[0] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    
                    # Post-process and draw bounding boxes with NMS
                    # Higher threshold (0.35) for faster inference and fewer boxes
                    boxes, scores = yolo_postprocess(outputs, conf_thresh=0.35, img_size=IMG_SIZE)
                    
                    # Apply Non-Maximum Suppression to remove duplicate detections
                    if boxes:
                        boxes, scores = nms(boxes, scores, iou_threshold=0.5)
                    
                    # Draw filtered bounding boxes
                    for box, score in zip(boxes, scores):
                        x1, y1, x2, y2 = box
                        # Draw bounding box
                        cv2.rectangle(img_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Draw label with confidence
                        label = f'Glasses {score:.2f}'
                        cv2.putText(img_disp, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.imshow(f'{MODEL_NAME} - Custom TFLite Model', img_disp)
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


