import subprocess
import os
import csv
import shutil
import sys
from datetime import datetime

# =========================
# CONFIGURATION
# =========================
DEPLOYED_MODEL = "models/deployed/best.pt"
CANDIDATE_MODEL = "models/candidates/best_ft/weights/best.pt"

HISTORY_FILE = "history/eval_history.csv"
LAST_EXPORT_FILE = "history/last_eval_export.txt"

MAP_DROP_THRESHOLD = -0.03
ABS_MAP_THRESHOLD = 0.60
IMPROVEMENT_MARGIN = 0.02

EPOCHS = 3
LR = 1e-4

# =========================
# UTILS
# =========================
def run(cmd, capture_output=True):
    """
    Executes a shell command.
    If capture_output is True, returns the stdout string.
    If capture_output is False, streams output to console (useful for training progress).
    """
    print(f"[CMD] {' '.join(cmd)}")
    
    if capture_output:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        if result.returncode != 0:
            print(f"[ERROR] Command failed: {result.stderr}")
            raise RuntimeError(result.stderr)
        return result.stdout
    else:
        # Stream output directly to console so user sees progress bars
        subprocess.run(cmd, check=True)
        return None


def extract_map50(output):
    """Parses YOLO validation output to find mAP50."""
    for line in output.splitlines():
        if "all" in line:
            parts = line.split()
            try:
                # YOLO output format usually puts mAP50 at index 5 or 6 depending on version
                # Attempting standard index 5 (Precision, Recall, mAP50, mAP50-95)
                return float(parts[5])
            except (IndexError, ValueError):
                continue
    raise RuntimeError("mAP50 not found in YOLO output")


def save_history(map50):
    """Logs the evaluation result to CSV."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    new_file = not os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "map50"])
        writer.writerow([datetime.now().isoformat(), map50])


def load_last_map50():
    """Retrieves the mAP50 from the previous run."""
    if not os.path.exists(HISTORY_FILE):
        return None
    with open(HISTORY_FILE, "r") as f:
        rows = list(csv.reader(f))
        if len(rows) < 2:
            return None
        return float(rows[-1][1])


def load_last_eval_export():
    """Gets the path of the dataset used in the previous run."""
    if not os.path.exists(LAST_EXPORT_FILE):
        return None
    with open(LAST_EXPORT_FILE, "r") as f:
        return f.read().strip()


def save_last_eval_export(path):
    """Saves the current dataset path for future reference."""
    os.makedirs(os.path.dirname(LAST_EXPORT_FILE), exist_ok=True)
    with open(LAST_EXPORT_FILE, "w") as f:
        f.write(path)

# =========================
# DATASET HANDLING
# =========================
def get_original_image_path():
    """Loops until a valid image path is provided."""
    while True:
        path = input("Enter Original Image Path: ").strip().strip('"')
        if os.path.exists(path):
            return path
        print(f"[ERROR] Path not found: {path}. Please try again.")


def handle_auto_export():
    """Runs the automated Label Studio export pipeline."""
    print("[INFO] Starting Auto-Export from Label Studio...")
    run(["python", "ls_export/export_yolo.py"])

    # Find the most recent export folder
    exports = sorted(
        [os.path.join("exports", d) for d in os.listdir("exports") if os.path.isdir(os.path.join("exports", d))],
        key=os.path.getmtime
    )
    
    if not exports:
        raise RuntimeError("No export directories found in 'exports/'")

    export_dir = exports[-1]
    print(f"[INFO] Using latest export: {export_dir}")

    # robustness: ensure path is valid
    original_img_path = get_original_image_path()

    run(["python", "ls_export/rebuild_yolo_dataset.py", export_dir, original_img_path])
    run(["python", "ls_export/create_yaml.py", export_dir])
    
    return export_dir


def handle_manual_export():
    """Handles user-provided dataset path."""
    while True:
        path = input("Enter path to existing YOLO dataset folder: ").strip().strip('"')
        if os.path.exists(path) and os.path.isdir(path):
            print(f"[INFO] Using manual export dataset: {path}")
            # We must run create_yaml to ensure 'data_eval.yaml' exists and points to this specific absolute path
            try:
                run(["python", "ls_export/create_yaml.py", path])
                return path
            except Exception as e:
                print(f"[ERROR] Failed to create YAML for this folder: {e}")
        else:
            print("[ERROR] Invalid directory. Please try again.")


def prepare_dataset():
    """Orchestrator for dataset preparation."""
    print("\n=== DATASET SELECTION ===")
    print("1. Auto Export (from Label Studio)")
    print("2. Manual Export (Existing Folder)")
    
    while True:
        choice = input("Select option (1/2): ").strip()
        if choice == '1':
            return handle_auto_export()
        elif choice == '2':
            return handle_manual_export()
        else:
            print("Invalid input. Please enter 1 or 2.")

# =========================
# EVALUATION & TRAINING
# =========================
def evaluate(model_path, dataset_dir):
    yaml_path = os.path.join(dataset_dir, "data_eval.yaml").replace("\\", "/")
    
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config not found at: {yaml_path}")

    print(f"[INFO] Evaluating model on: {dataset_dir}")
    output = run([
        "yolo",
        "task=detect",
        "mode=val",
        f"model={model_path}",
        f"data={yaml_path}",
        "imgsz=640",
        "verbose=False"
    ])
    return extract_map50(output)


def check_finetune_recommendation(current_map50):
    """Returns (bool, str) - (Should Finetune, Reason)"""
    last_map50 = load_last_map50()

    if last_map50 is None:
        return False, "First run (baseline established)"

    delta = current_map50 - last_map50
    print(f"[INFO] ΔmAP50 = {delta:.4f}")

    if current_map50 < ABS_MAP_THRESHOLD:
        return True, f"Current mAP50 ({current_map50:.4f}) is below absolute threshold ({ABS_MAP_THRESHOLD})"

    if delta < MAP_DROP_THRESHOLD:
        return True, f"Performance dropped by {delta:.4f} (Threshold: {MAP_DROP_THRESHOLD})"

    return False, "Performance is stable"


def fine_tune(previous_export, current_export):
    """
    Runs fine-tuning. 
    Uses previous_export (verified data) mixed with new data if logic allows,
    or falls back to current export if previous is missing.
    """
    os.makedirs("models/candidates", exist_ok=True)
    
    # If this is a fresh run with no history, we might not have a previous export.
    # In that case, we use the current export as the base.
    base_data = previous_export if previous_export else current_export
    
    print(f"[INFO] Building fine-tune dataset based on: {base_data}")

    run([
        "python",
        "ls_export/build_fine_tune_dataset.py",
        base_data
    ])

    print("[INFO] Starting Training...")
    # Training takes time, so we set capture_output=False to show the YOLO progress bar
    run([
        "yolo",
        "task=detect",
        "mode=train",
        f"model={DEPLOYED_MODEL}",
        "data=finetune_dataset/data_finetune.yaml",
        f"epochs={EPOCHS}",
        f"lr0={LR}",
        "imgsz=640",
        f"project=models/candidates",
        "name=best_ft",
        "exist_ok=True"
    ], capture_output=False)


def compare_and_promote(eval_export):
    print("\n[STEP] Comparing deployed vs candidate model")

    map_old = evaluate(DEPLOYED_MODEL, eval_export)
    map_new = evaluate(CANDIDATE_MODEL, eval_export)

    print(f"[RESULT] Deployed mAP50 : {map_old:.4f}")
    print(f"[RESULT] Candidate mAP50: {map_new:.4f}")

    if map_new > map_old + IMPROVEMENT_MARGIN:
        print("[PROMOTION] New model is significantly better → deploying")
        shutil.copy(CANDIDATE_MODEL, DEPLOYED_MODEL)
        return True

    print("[INFO] Candidate not better (within margin) → keeping deployed model")
    return False

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    try:
        # 1. Prepare Data
        export_dir = prepare_dataset()

        # 2. Evaluate
        print("\n[STEP] Evaluate deployed model")
        map50 = evaluate(DEPLOYED_MODEL, export_dir)
        save_history(map50)
        print(f"[RESULT] Current mAP50 = {map50:.4f}")

        # 3. Analyze & Ask User
        previous_export = load_last_eval_export()
        
        # Save current as last for next time
        save_last_eval_export(export_dir)

        should_ft, reason = check_finetune_recommendation(map50)
        
        print("\n=== PIPELINE DECISION ===")
        print(f"System Recommendation: {'FINE-TUNE' if should_ft else 'DO NOT FINE-TUNE'}")
        print(f"Reason: {reason}")
        print("The fine-tune procees will take long time. Ensure your device in the stable state")
        
        user_choice = input("Do you want to proceed with fine-tuning? (y/n): ").strip().lower()

        if user_choice == 'y':
            print("\n[STEP] Starting Fine-tuning process...")
            fine_tune(previous_export, export_dir)

            print("\n[STEP] Verification & Promotion")
            compare_and_promote(export_dir)
        else:
            print("\n[DONE] Pipeline stopped by user.")
            
    except KeyboardInterrupt:
        print("\n[ABORT] Operation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        sys.exit(1)