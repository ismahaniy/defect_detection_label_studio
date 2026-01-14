import subprocess
import shutil
import os

# =========================
# CONFIGURATION
# =========================
EVAL_IOU = "0.5"
IMPROVEMENT_MARGIN = 0.02   # +2% mAP50 required

DEPLOYED_MODEL = "models/deployed/best.pt"
CANDIDATE_MODEL = "models/candidates/best_ft/weights/best.pt"

# =========================
# UTILS
# =========================
def run(cmd):
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout


def extract_map50(output):
    for line in output.splitlines():
        if "all" in line:
            parts = line.split()
            try:
                return float(parts[5])
            except (IndexError, ValueError):
                continue
    raise RuntimeError("mAP50 not found")


def evaluate_model(model_path, data_yaml):
    print(f"[EVAL] Evaluating {model_path}")
    output = run([
        "yolo",
        "task=detect",
        "mode=val",
        f"model={model_path}",
        f"data={data_yaml}",
        f"iou={EVAL_IOU}",
        "verbose=False"
    ])
    return extract_map50(output)


# =========================
# COMPARISON LOGIC
# =========================
def compare_and_promote(data_yaml):
    if not os.path.exists(DEPLOYED_MODEL):
        raise RuntimeError("No deployed model found")

    if not os.path.exists(CANDIDATE_MODEL):
        print("[INFO] No candidate model available. Skipping comparison.")
        return False

    map_old = evaluate_model(DEPLOYED_MODEL, data_yaml)
    map_new = evaluate_model(CANDIDATE_MODEL, data_yaml)

    print(f"[RESULT] Deployed mAP50:  {map_old:.4f}")
    print(f"[RESULT] Candidate mAP50: {map_new:.4f}")

    if map_new > map_old + IMPROVEMENT_MARGIN:
        print("[PROMOTION] Candidate model is better. Deploying...")
        shutil.copy(CANDIDATE_MODEL, DEPLOYED_MODEL)
        print("[DONE] New model deployed.")
        return True

    print("[INFO] Candidate model not better. Keeping deployed model.")
    return False


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    data_yaml = input("Enter NEW evaluation data_eval.yaml path: ").strip()
    compare_and_promote(data_yaml)
