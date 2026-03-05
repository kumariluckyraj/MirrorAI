# baseline_storage.py
# Manages personal baseline — created ONCE from Day 1 session average.
# Never overwritten unless explicit recalibration is triggered.

import json
import os
from datetime import datetime

BASELINE_PATH = "data/user_baseline.json"
TEMP_BUFFER_PATH = "data/baseline_buffer.json"


def baseline_exists():
    """Returns True if a finalized personal baseline exists."""
    return os.path.exists(BASELINE_PATH)


def is_collecting():
    """
    Returns True if a baseline collection session is in progress.
    This means Day 1 session was interrupted — buffer exists but
    baseline was never finalized.
    """
    return os.path.exists(TEMP_BUFFER_PATH) and not baseline_exists()


def push_to_buffer(features: dict):
    """
    During Day 1 session, call this every N frames.
    Appends feature snapshot to temporary buffer.
    Buffer is stored on disk so interrupted sessions can resume.

    Parameters:
        features (dict): extracted feature values for this frame
    """
    os.makedirs("data", exist_ok=True)

    # Load existing buffer or start fresh
    if os.path.exists(TEMP_BUFFER_PATH):
        with open(TEMP_BUFFER_PATH, "r") as f:
            buffer = json.load(f)
    else:
        buffer = {"samples": [], "started_at": datetime.now().isoformat()}

    buffer["samples"].append(features)
    buffer["sample_count"] = len(buffer["samples"])

    with open(TEMP_BUFFER_PATH, "w") as f:
        json.dump(buffer, f, indent=4)


def finalize_baseline():
    """
    Called at END of Day 1 session.
    Averages all buffered feature samples and saves as permanent baseline.
    Deletes the temporary buffer after saving.

    Returns:
        dict — the finalized baseline, or None if buffer was empty/insufficient
    """
    if not os.path.exists(TEMP_BUFFER_PATH):
        print("[baseline_storage] No buffer found. Cannot finalize.")
        return None

    with open(TEMP_BUFFER_PATH, "r") as f:
        buffer = json.load(f)

    samples = buffer.get("samples", [])

    # Edge case — not enough data collected
    if len(samples) < 10:
        print(f"[baseline_storage] Only {len(samples)} samples collected. Need at least 10. Discarding.")
        os.remove(TEMP_BUFFER_PATH)
        return None

    print(f"[baseline_storage] Finalizing baseline from {len(samples)} samples...")

    # Average all flat features
    flat_keys = ["A_mean", "B_mean", "lip_dryness", "eye_darkness", "BPM"]
    baseline = {}

    for key in flat_keys:
        values = [s[key] for s in samples if key in s]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
        std = max(variance ** 0.5, 0.001)
        baseline[key] = {"mean": round(mean, 6), "std": round(std, 6)}

    # Average skin features — always stored nested
    skin_keys = ["mole_asymmetry", "mole_border", "mole_color_var"]
    baseline["skin"] = {}

    for key in skin_keys:
        values = [s[key] for s in samples if key in s]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
        std = max(variance ** 0.5, 0.001)
        baseline["skin"][key] = {"mean": round(mean, 6), "std": round(std, 6)}

    # Metadata
    baseline["created_at"] = datetime.now().isoformat()
    baseline["sample_count"] = len(samples)
    baseline["started_at"] = buffer.get("started_at", "unknown")
    baseline["version"] = 1

    # Save permanently
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=4)

    # Delete temp buffer
    os.remove(TEMP_BUFFER_PATH)

    print(f"[baseline_storage] Baseline finalized and saved → {BASELINE_PATH}")
    print(f"[baseline_storage] A_mean baseline: {baseline['A_mean']['mean']:.2f} ± {baseline['A_mean']['std']:.2f}")
    return baseline


def load_baseline():
    """
    Loads the finalized personal baseline.
    Raises FileNotFoundError if not found — always call baseline_exists() first.
    """
    if not baseline_exists():
        raise FileNotFoundError(
            f"No personal baseline at {BASELINE_PATH}. Complete a Day 1 session first."
        )
    with open(BASELINE_PATH, "r") as f:
        baseline = json.load(f)
    print(f"[baseline_storage] Loaded personal baseline "
          f"(created: {baseline.get('created_at', 'unknown')}, "
          f"samples: {baseline.get('sample_count', '?')})")
    return baseline


def recalibrate():
    """
    EXPLICIT recalibration only. Deletes personal baseline so next
    session runs as Day 1 again.
    Call this only when user explicitly requests recalibration.
    """
    if os.path.exists(BASELINE_PATH):
        os.remove(BASELINE_PATH)
        print("[baseline_storage] Personal baseline deleted. Next session = Day 1.")
    if os.path.exists(TEMP_BUFFER_PATH):
        os.remove(TEMP_BUFFER_PATH)
        print("[baseline_storage] Temp buffer cleared.")


def get_buffer_status():
    """
    Returns how many samples have been collected so far in Day 1 session.
    Useful for showing progress bar on screen.
    """
    if not os.path.exists(TEMP_BUFFER_PATH):
        return 0
    with open(TEMP_BUFFER_PATH, "r") as f:
        buffer = json.load(f)
    return buffer.get("sample_count", 0)