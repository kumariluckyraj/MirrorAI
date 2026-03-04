import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# Left eye landmarks
LEFT_EYE_UPPER = [159, 160, 161]
LEFT_EYE_LOWER = [145, 144, 163]

# Right eye landmarks
RIGHT_EYE_UPPER = [386, 387, 388]
RIGHT_EYE_LOWER = [374, 380, 381]

cap = cv2.VideoCapture(0)

# For smoothing
risk_history = []

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                all_eye_points = []

                # LEFT EYE
                for idx in LEFT_EYE_UPPER + LEFT_EYE_LOWER:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    all_eye_points.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # RIGHT EYE
                for idx in RIGHT_EYE_UPPER + RIGHT_EYE_LOWER:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    all_eye_points.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

                pts = np.array(all_eye_points)

                x_min = np.min(pts[:, 0])
                x_max = np.max(pts[:, 0])
                y_min = np.min(pts[:, 1])
                y_max = np.max(pts[:, 1])

                pad = 5
                x_min = max(0, x_min - pad)
                y_min = max(0, y_min - pad)
                x_max = min(w, x_max + pad)
                y_max = min(h, y_max + pad)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                eyelid_crop = frame[y_min:y_max, x_min:x_max]

                if eyelid_crop.size != 0:

                    lab = cv2.cvtColor(eyelid_crop, cv2.COLOR_BGR2LAB)
                    A_channel = lab[:, :, 1]
                    A_mean = np.mean(A_channel)

                    # ----- Risk Calculation -----
                    min_val = 130
                    max_val = 170

                    score = (A_mean - min_val) / (max_val - min_val)
                    score = max(0, min(score, 1))

                    risk_percentage = int((1 - score) * 100)

                    # ----- Smoothing -----
                    risk_history.append(risk_percentage)
                    if len(risk_history) > 10:
                        risk_history.pop(0)

                    smooth_risk = int(np.mean(risk_history))

                    # ----- Risk Color -----
                    if smooth_risk < 30:
                        color = (0, 255, 0)  # Green
                        status = "Low Risk"
                    elif smooth_risk < 60:
                        color = (0, 255, 255)  # Yellow
                        status = "Moderate Risk"
                    else:
                        color = (0, 0, 255)  # Red
                        status = "High Risk"

                    # ----- Display -----
                    cv2.putText(frame,
                                f"Anemia Risk: {smooth_risk}%",
                                (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                color,
                                2)

                    cv2.putText(frame,
                                status,
                                (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                color,
                                2)

        cv2.imshow("Health Mirror - Anemia Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()