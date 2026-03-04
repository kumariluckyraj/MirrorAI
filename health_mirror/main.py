# from video_capture import capture_video

# frames = capture_video(10)
# print(len(frames))

import cv2
from video_capture import capture_video
from face_detection import detect_landmarks
import mediapipe as mp
print("Script started")
mp_drawing = mp.solutions.drawing_utils

frames = capture_video(5)

for frame in frames:
    landmarks = detect_landmarks(frame)
    if landmarks:
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp.solutions.face_mesh.FACEMESH_TESSELATION
        )
    cv2.imshow("Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()