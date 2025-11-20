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
SHOW_DISPLAY = True  # Set to False if running headless (no display)

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
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("❌ Error: Could not open camera!")
        return

    log_path = f"logs/{model_name}.csv"
    start_time = time.time()
    frame_count = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "fps", "cpu", "ram", "temp", "detections"])

        while time.time() - start_time < TEST_DURATION:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame from camera")
                break
            
            frame_count += 1
            t0 = time.time()
            results = model(frame, imgsz=IMG_SIZE, verbose=False)
            inference_time = time.time() - t0
            fps = 1 / inference_time

            # Draw bounding boxes and labels on frame
            annotated_frame = results[0].plot()
            
            # Count detections
            detections = len(results[0].boxes)
            
            # Get detected class names
            detected_objects = []
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]
                conf = float(box.conf[0])
                detected_objects.append(f"{class_name}({conf:.2f})")
            
            cpu, ram, temp = get_system_stats()
            writer.writerow([frame_count, fps, cpu, ram, temp, detections])

            detection_str = ", ".join(detected_objects) if detected_objects else "None"
            print(f"{model_name} | Frame: {frame_count} | FPS: {fps:.2f} | CPU: {cpu}% | RAM: {ram}% | Temp: {temp}°C | Detections: {detections} | Objects: [{detection_str}]")
            
            # Display live view if display is available
            if SHOW_DISPLAY:
                try:
                    cv2.imshow(f'{model_name} - YOLO Live Detection', annotated_frame)
                    # Press 'q' to quit early
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⏹️  Stopped by user")
                        break
                except:
                    # If display fails, disable it
                    SHOW_DISPLAY = False
                    print("⚠️  Display not available, running in headless mode")

    cap.release()
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()
    
    print(f"\n✅ Test complete for {model_name}! Results saved to {log_path}")


if __name__ == "__main__":
    for name, path in MODELS.items():
        run_single_test(name, path)
