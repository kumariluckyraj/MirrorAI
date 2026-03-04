import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

def detect_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]
    return None


def get_landmark_coords(frame, landmarks):
    h, w, _ = frame.shape
    coords = []

    for lm in landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        coords.append((x, y))

    return coords