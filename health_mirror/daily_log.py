# daily_log.py
import json
import os
from datetime import datetime

LOG_DIR = "data/logs"


def log_session(features: dict, risks: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"{today}.jsonl")
    entry = {
        "timestamp": datetime.now().isoformat(),
        "features": features,
        "risks": risks
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[daily_log] Session logged → {log_path}")


def load_today_log():
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"{today}.jsonl")
    if not os.path.exists(log_path):
        return []
    entries = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_log_for_date(date_str: str):
    log_path = os.path.join(LOG_DIR, f"{date_str}.jsonl")
    if not os.path.exists(log_path):
        return []
    entries = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def get_all_log_dates():
    if not os.path.exists(LOG_DIR):
        return []
    return sorted([
        f.replace(".jsonl", "")
        for f in os.listdir(LOG_DIR)
        if f.endswith(".jsonl")
    ])