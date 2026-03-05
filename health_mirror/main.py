# main.py — MirrorAI Full Pipeline
# Day 1: Baseline Collection Mode — population risk only, collect features
# Day 2+: Hybrid Mode — population + personal deviation combined

import cv2
import json
import numpy as np
from collections import deque
from face_detection import detect_landmarks, get_landmark_coords
from risk_engine import compute_all_risks
from baseline_storage import (
    baseline_exists, load_baseline,
    push_to_buffer, finalize_baseline,
    get_buffer_status, is_collecting
)
from daily_log import log_session
from modules.anemia import extract_eyelid_region, compute_anemia_score

# ── Load population baseline (always loaded) ───────────────────
with open("data/dataset_baseline.json", "r") as f:
    pop_baseline = json.load(f)

# ── Determine session mode ─────────────────────────────────────
if baseline_exists():
    personal_baseline = load_baseline()
    session_mode = "HYBRID"
    print("[main] Day 2+ mode. Personal baseline loaded.")
else:
    personal_baseline = None
    session_mode = "COLLECTING"
    print("[main] Day 1 mode. Collecting baseline this session.")
    if is_collecting():
        print("[main] Resuming interrupted Day 1 session.")

# ── Smoothing buffer for A_mean ────────────────────────────────
a_mean_buffer = deque(maxlen=15)

# ── Frame counters ─────────────────────────────────────────────
frame_count = 0
SAMPLE_EVERY = 30

# ── Open webcam ────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[main] Camera failed. Exiting.")
    exit()

print("[main] Camera open. Press ESC to exit.")

# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════
while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = detect_landmarks(frame)

    if landmarks:
        coords = get_landmark_coords(frame, landmarks)

        # ── Feature extraction ─────────────────────────────────
        eyelid = extract_eyelid_region(frame, coords)
        raw_A = compute_anemia_score(eyelid)

        if raw_A and raw_A > 0:
            a_mean_buffer.append(raw_A)

        A_mean = float(np.mean(a_mean_buffer)) if a_mean_buffer else 150.0

        features = {
            "A_mean":         A_mean,
            "B_mean":         132.0,    # TODO: P2
            "lip_dryness":    0.3,      # TODO: P2
            "eye_darkness":   0.27,     # TODO: P2
            "BPM":            75.0,     # TODO: P2
            "mole_asymmetry": 0.17,     # TODO: P2
            "mole_border":    0.20,     # TODO: P2
            "mole_color_var": 10.4      # TODO: P2
        }

        # ── Risk calculation ───────────────────────────────────
        risks = compute_all_risks(features, pop_baseline, personal_baseline)

        frame_count += 1

        # ── Buffer sampling (Day 1 only) ───────────────────────
        if session_mode == "COLLECTING" and frame_count % SAMPLE_EVERY == 0:
            push_to_buffer(features)
            samples = get_buffer_status()
            print(f"[main] Collecting... {samples} samples saved")

        # ── Session logging (Day 2+ only) ──────────────────────
        if session_mode == "HYBRID" and frame_count % SAMPLE_EVERY == 0:
            log_session(features, risks)

        # ══════════════════════════════════════════════════════
        # DISPLAY
        # ══════════════════════════════════════════════════════
        # Background panel
        cv2.rectangle(frame, (10, 10), (340, 310), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (340, 310), (60, 60, 60), 1)

        # Mode banner
        if session_mode == "COLLECTING":
            samples = get_buffer_status()
            banner = f"DAY 1 - COLLECTING BASELINE ({samples} samples)"
            banner_color = (0, 200, 255)
        else:
            banner = "HYBRID MODE - Day 2+"
            banner_color = (0, 255, 150)

        cv2.putText(frame, banner, (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, banner_color, 1)

        # Divider
        cv2.line(frame, (18, 38), (330, 38), (60, 60, 60), 1)

        # Risk scores
        y = 58
        for condition, score in risks.items():
            if score < 30:
                color = (0, 255, 0)
                tag = "LOW"
            elif score < 60:
                color = (0, 255, 255)
                tag = "MOD"
            else:
                color = (0, 0, 255)
                tag = "HIGH"

            if session_mode == "COLLECTING":
                color = tuple(int(c * 0.7) for c in color)

            cv2.putText(frame,
                        f"{condition:<20} {score:>5.1f}%  {tag}",
                        (18, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, color, 1)
            y += 26

        # A_mean and frame info
        cv2.putText(frame,
                    f"A_mean: {A_mean:.1f}   Frame: {frame_count}",
                    (18, y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (150, 150, 150), 1)

    else:
        cv2.rectangle(frame, (10, 10), (300, 50), (0, 0, 0), -1)
        cv2.putText(frame, "No face detected",
                    (18, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    cv2.imshow("MirrorAI Health Mirror", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ══════════════════════════════════════════════════════════════
# SESSION END
# ══════════════════════════════════════════════════════════════
cap.release()
cv2.destroyAllWindows()

if session_mode == "COLLECTING":
    print("[main] Session ended. Finalizing baseline...")
    result = finalize_baseline()
    if result:
        print(f"[main] Baseline saved from {result['sample_count']} samples.")
        print(f"[main] Your A_mean baseline: {result['A_mean']['mean']:.2f}")
        print("[main] Next session will use HYBRID mode.")
    else:
        print("[main] Not enough data. Run again to complete Day 1.")
else:
    print("[main] Session ended.")