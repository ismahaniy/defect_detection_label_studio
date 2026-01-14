import os
import shutil
import random
import yaml
import sys

# =========================
# CONFIGURATION
# =========================
CLASS_NAMES = ["dentado", "concave", "perforation"]
OLD_TRAIN_DIR = "yolo_dataset_defect_detection"
OUTPUT_DIR = "finetune_dataset"

IMAGE_EXTS = [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]

def list_image_label_pairs(base_dir):
    """
    Search recursively for image/label pairs to handle subfolders 
    like 'train' and 'val' within the directory.
    """
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    pairs = []

    if not os.path.exists(labels_dir):
        print(f"[DEBUG] Labels directory not found at: {labels_dir}")
        return []

    # Use os.walk to find all .txt files in all subdirectories of labels
    for root, dirs, files in os.walk(labels_dir):
        for lbl in files:
            if not lbl.endswith(".txt") or lbl == "classes.txt":
                continue

            # Get filename without extension
            name = os.path.splitext(lbl)[0]
            
            # Find the relative path from the labels_dir to the current folder
            # This helps us look in the corresponding 'images' subfolder
            rel_path = os.path.relpath(root, labels_dir)
            target_img_subdir = os.path.join(images_dir, rel_path)

            for ext in IMAGE_EXTS:
                img_path = os.path.join(target_img_subdir, name + ext)
                if os.path.exists(img_path):
                    pairs.append((img_path, os.path.join(root, lbl)))
                    break
    
    print(f"[DEBUG] Found {len(pairs)} pairs in {base_dir}")
    return pairs

def copy_pairs(pairs, out_images, out_labels):
    copied_count = 0
    for img, lbl in pairs:
        dest_img = os.path.join(out_images, os.path.basename(img))
        dest_lbl = os.path.join(out_labels, os.path.basename(lbl))
        
        # Avoid overwriting or duplicate work if file already exists
        if not os.path.exists(dest_img):
            shutil.copy(img, dest_img)
            shutil.copy(lbl, dest_lbl)
            copied_count += 1
    return copied_count

# =========================
# MAIN LOGIC
# =========================
def build_finetune_dataset(new_export_dir):
    out_images = os.path.join(OUTPUT_DIR, "images")
    out_labels = os.path.join(OUTPUT_DIR, "labels")

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    # We only take a portion of old data to mix in if this is the FIRST time building
    # Otherwise, we just keep adding new verified data to the pool
    old_pairs = list_image_label_pairs(OLD_TRAIN_DIR)
    random.shuffle(old_pairs)
    selected_old = old_pairs[:min(len(old_pairs), 50)] # Grab 50 representative old samples

    print(f"[INFO] Collecting new data from: {new_export_dir}")
    new_pairs = list_image_label_pairs(new_export_dir)
    

    # Copy data (Append mode)
    added_new = copy_pairs(new_pairs, out_images, out_labels)
    added_old = copy_pairs(selected_old, out_images, out_labels)

    print(f"[SUCCESS] Added {added_new} new verified images.")
    print(f"[SUCCESS] Ensured {added_old} old representative images are present.")

    # Create the training YAML
    create_finetune_yaml()

def create_finetune_yaml():
    abs_out_dir = os.path.abspath(OUTPUT_DIR).replace("\\", "/")
    data = {
        "path": abs_out_dir,
        "train": "images",
        "val": "images", # During fine-tune, we monitor against the growing pool
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }
    
    yaml_path = "finetune_dataset/data_finetune.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"[INFO] Created {yaml_path}")

if __name__ == "__main__":
    # Get export dir from pipeline argument
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = input("Enter latest export directory: ").strip().strip('"')
        
    build_finetune_dataset(target_dir)
