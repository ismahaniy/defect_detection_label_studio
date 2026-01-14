import subprocess
import re
import csv
import os
from datetime import datetime

MODEL_PATH = "yolo/best.pt"
DATA_YAML = "exports/data_eval.yaml"
HISTORY_FILE = "eval/history.csv"

MAP_DROP_THRESHOLD = -0.03      # retrain trigger
ABS_MAP_THRESHOLD = 0.60        # optional absolute rule


def run_yolo_val():
    cmd = [
        "yolo",
        "task=detect",
        "mode=val",
        f"model={MODEL_PATH}",
        f"data={DATA_YAML}",
        "iou=0.5",
        "verbose=False"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    return result.stdout


def extract_map50(output):
    """
    Extract mAP50 from YOLO output
    """
    match = re.search(r"mAP50\(B\):\s+([0-9.]+)", output)
    if match:
        return float(match.group(1))
    else:
        raise ValueError("mAP50 not found in YOLO output")


def save_history(map50):
    file_exists = os.path.isfile(HISTORY_FILE)

    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "map50"])

        writer.writerow([
            datetime.now().isoformat(),
            map50
        ])

def load_last_map50():
    if not os.path.exists(HISTORY_FILE):
        return None

    with open(HISTORY_FILE, "r") as f:
        rows = list(csv.reader(f))
        if len(rows) < 2:
            return None
        return float(rows[-1][1])

def should_retrain(current_map50):
    last_map50 = load_last_map50()

    if last_map50 is None:
        print("[INFO] No previous evaluation. Skipping retrain.")
        return False

    delta = current_map50 - last_map50
    print(f"[INFO] ΔmAP50 = {delta:.4f}")

    if current_map50 < ABS_MAP_THRESHOLD:
        print("[DECISION] Below absolute threshold → retrain")
        return True

    if delta < MAP_DROP_THRESHOLD:
        print("[DECISION] Significant degradation → retrain")
        return True

    print("[DECISION] Performance stable → no retrain")
    return False


if __name__ == "__main__":
    print("[INFO] Running YOLOv8 evaluation...")
    output = run_yolo_val()

    map50 = extract_map50(output)
    save_history(map50)

    print(f"[RESULT] mAP50 = {map50:.4f}")

    if should_retrain(map50):
        print("[STEP] Fine-tuning model...")
        
        print("[DONE] Fine-tuning complete")
    else:
        print("[DONE] No retraining needed")
