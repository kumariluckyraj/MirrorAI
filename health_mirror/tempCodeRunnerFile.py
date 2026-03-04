import cv2
from video_capture import capture_video
from face_detection import detect_landmarks, get_landmark_coords
from modules.anemia import extract_eyelid_region, compute_anemia_score

frames = capture_video(5)

for frame in frames:
    landmarks = detect_landmarks(frame)

    if landmarks:
        coords = get_landmark_coords(frame, landmarks)

        eyelid = extract_eyelid_region(frame, coords)

        if eyelid is not None and eyelid.size != 0:
            score = compute_anemia_score(eyelid)
            print("A_mean:", score)

            cv2.imshow("Eyelid Region", eyelid)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
