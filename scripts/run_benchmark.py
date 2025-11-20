from ultralytics import YOLO
import time, csv, cv2, os, sys
from system_monitor import get_system_stats
from camera_stream import get_camera_stream

# Change to project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

TEST_DURATION = 30  # seconds
IMG_SIZE = 320

MODELS = {
    "yolo8n": os.path.join(project_root, "models/yolov8n.pt"),
    "yolo11n": os.path.join(project_root, "models/yolo11n.pt")
}

def run_single_test(model_name, model_path):
    print(f"\n=== Running test for {model_name} ===\n")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print(f"Downloading {model_name}...")
    
    model = YOLO(model_path)
    cap = get_camera_stream()

    log_path = f"logs/{model_name}.csv"
    start_time = time.time()

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fps", "cpu", "ram", "temp", "detections"])

        while time.time() - start_time < TEST_DURATION:
            ret, frame = cap.read()
            if not ret:
                break
            
            t0 = time.time()
            results = model(frame, imgsz=IMG_SIZE, verbose=False)
            fps = 1 / (time.time() - t0)

            # Draw bounding boxes and labels on frame
            annotated_frame = results[0].plot()
            
            # Count detections
            detections = len(results[0].boxes)
            
            # Display live view
            cv2.imshow(f'{model_name} - YOLO Live Detection', annotated_frame)
            
            cpu, ram, temp = get_system_stats()
            writer.writerow([fps, cpu, ram, temp, detections])

            print(f"{model_name} | FPS: {fps:.2f} | CPU: {cpu}% | RAM: {ram}% | Temp: {temp}°C | Objects: {detections}")
            
            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    for name, path in MODELS.items():
        run_single_test(name, path)
