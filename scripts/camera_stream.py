import cv2

def get_camera_stream(width=320, height=240):
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)
    return cap
