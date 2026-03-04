import cv2
import time

def capture_video(duration=10):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Camera not accessible")
        return None
    
    frames = []
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    print(f"Captured {len(frames)} frames")
    return frames