from ultralytics import YOLO
import time, csv, cv2
from system_monitor import get_system_stats
from camera_stream import get_camera_stream

TEST_DURATION = 30  # seconds
IMG_SIZE = 320

MODELS = {
    "yolo8n": "models/yolov8n.pt",
    "yolo11n": "models/yolov11n.pt"
}

def run_single_test(model_name, model_path):
    print(f"\n=== Running test for {model_name} ===\n")
    
    model = YOLO(model_path)
    cap = get_camera_stream()

    log_path = f"logs/{model_name}.csv"
    start_time = time.time()

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fps", "cpu", "ram", "temp"])

        while time.time() - start_time < TEST_DURATION:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            results = model(frame, imgsz=IMG_SIZE)
            fps = 1 / (time.time() - t0)

            cpu, ram, temp = get_system_stats()
            writer.writerow([fps, cpu, ram, temp])

            print(f"{model_name} | FPS: {fps:.2f} | CPU: {cpu}% | RAM: {ram}% | Temp: {temp}°C")

    cap.release()


if __name__ == "__main__":
    for name, path in MODELS.items():
        run_single_test(name, path)
