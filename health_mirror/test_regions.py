import cv2
from video_capture import capture_video
from face_detection import detect_landmarks, get_landmark_coords

frames = capture_video(5)

for frame in frames:
    landmarks = detect_landmarks(frame)

    if landmarks:
        coords = get_landmark_coords(frame, landmarks)

        # Example cheek landmarks (approximate region)
        x1, y1 = coords[234]  # left cheek area
        x2, y2 = coords[454]  # right cheek area

        # Draw rectangle around middle cheek zone
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cv2.rectangle(frame,
                      (cx - 50, cy - 30),
                      (cx + 50, cy + 30),
                      (0, 255, 0),
                      2)

    cv2.imshow("Cheek Region Test", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()