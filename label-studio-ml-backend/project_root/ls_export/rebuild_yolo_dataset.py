import os
import shutil
import requests
import sys
import argparse

# =========================
# CONFIGURATION
# =========================
LABEL_STUDIO_URL = "http://localhost:8080"
API_KEY = "62bfaba676e48bfe68449aa9580d3bf12e4b3c7a"
PROJECT_ID = 1  

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_dir", help="Path to LS export")
    parser.add_argument("--original_dir", help="Path to original images")
    return parser.parse_args()

# =========================
# STEP 1: Fetch task metadata
# =========================
def fetch_tasks(headers):
    url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/tasks"
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict) and "results" in data:
            return data["results"]
        if isinstance(data, list):
            return data
        raise RuntimeError("Unexpected Label Studio task response format")
    except Exception as e:
        print(f"[ERROR] Could not connect to Label Studio: {e}")
        return []

# =========================
# STEP 2 & 3: Align & Cleanup
# =========================
def process_dataset(export_dir, original_image_dir):
    # Setup paths based on export dir
    label_dir = os.path.join(export_dir, "labels")
    output_image_dir = os.path.join(export_dir, "images")
    
    if not os.path.exists(label_dir):
        print(f"[ERROR] Label directory not found: {label_dir}")
        return

    os.makedirs(output_image_dir, exist_ok=True)

    labels_processed = 0
    images_found = 0
    labels_deleted = 0

    print(f"[INFO] Processing labels in: {label_dir}")

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt") or label_file == "classes.txt":
            continue

        labels_processed += 1
        name_no_ext = os.path.splitext(label_file)[0]

        # Extract original filename (handles LS UUID prefix: 'uuid-filename.txt')
        if "-" in name_no_ext:
            original_name = name_no_ext.split("-", 1)[1]
        else:
            original_name = name_no_ext

        found = False
        # Check for common image extensions
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
            src_image = os.path.join(original_image_dir, original_name + ext)

            if os.path.exists(src_image):
                dst_image = os.path.join(output_image_dir, name_no_ext + ext)
                shutil.copy(src_image, dst_image)
                found = True
                images_found += 1
                break

        # If no image was found, delete the label file
        if not found:
            label_path = os.path.join(label_dir, label_file)
            try:
                os.remove(label_path)
                print(f"[CLEANUP] Deleted label (no image found): {label_file}")
                labels_deleted += 1
            except Exception as e:
                print(f"[ERROR] Failed to delete {label_file}: {e}")

    print("\n--- Summary ---")
    print(f"Total labels checked: {labels_processed}")
    print(f"Images matched/copied: {images_found}")
    print(f"Orphan labels deleted: {labels_deleted}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        EXPORT_DIR = sys.argv[1]
        ORIGINAL_IMAGE_DIR = sys.argv[2]
    else:
        EXPORT_DIR = input("Export Path: ")
        ORIGINAL_IMAGE_DIR = input("Original Path: ")

    process_dataset(EXPORT_DIR, ORIGINAL_IMAGE_DIR)
    print("\n[SUCCESS] Dataset alignment complete.")