import os
import yaml
import sys

# =========================
# CONFIGURATION
# =========================
CLASS_NAMES = ["dentado", "concave", "perforation"]

# =========================
# FUNCTION
# =========================
def create_data_yaml(export_dir):
    # Standardize path
    export_dir = os.path.abspath(export_dir)
    images_dir = os.path.join(export_dir, "images")
    labels_dir = os.path.join(export_dir, "labels")

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        raise RuntimeError(
            f"Invalid YOLO export folder: {export_dir}\n"
            f"Expected 'images/' and 'labels/' subfolders."
        )

    data = {
        "path": export_dir.replace("\\", "/"),
        "train": "images",
        "val": "images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }

    yaml_path = os.path.join(export_dir, "data_eval.yaml")

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"[SUCCESS] data_eval.yaml created at: {yaml_path}")


# =========================
# MAIN (Argument Friendly)
# =========================
if __name__ == "__main__":
    # Check if path was passed as an argument, otherwise fallback to input
    if len(sys.argv) > 1:
        dir_path = sys.argv[1]
    else:
        dir_path = input("Enter YOLO export folder path: ").strip().strip('"')
    
    create_data_yaml(dir_path)