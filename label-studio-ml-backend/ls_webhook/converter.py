import os
import requests
from PIL import Image
from urllib.parse import urlparse

LABEL_STUDIO_BASE_URL = "http://localhost:8080"
LABEL_STUDIO_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3MjU0NDU1NSwiaWF0IjoxNzY1MzQ0NTU1LCJqdGkiOiIxZjkyODI5NjNlZTg0YzMxYWZlOGJmNDkxOTQxZmUxNSIsInVzZXJfaWQiOiIxIn0.fHHOl-E_hQDP_ZU4zx60_Bsqbsqf0lSdjafoX7pxvv0"

CLASS_MAP = {
    "dentado": 0,
    "concave": 1,
    "perforation": 2
}

DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "images")
LBL_DIR = os.path.join(DATA_DIR, "labels")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)


"""def download_image(url, task_id):
    if url.startswith("/"):
        url = LABEL_STUDIO_BASE_URL + url

    headers = {
        "Authorization": LABEL_STUDIO_API_KEY
    }

    filename = f"{task_id}.jpg"
    path = os.path.join(IMG_DIR, filename)

    if not os.path.exists(path):
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)

    return path
"""





def convert_ls_to_yolo(payload):
    task = payload["task"]
    annotation = payload["annotation"]

    task_id = task["id"]
    label_path = os.path.join(LBL_DIR, f"{task_id}.txt")

    yolo_lines = []

    for result in annotation["result"]:
        if result["type"] != "rectanglelabels":
            continue

        value = result["value"]
        label_name = value["rectanglelabels"][0]
        class_id = CLASS_MAP[label_name]

        x = value["x"] / 100
        y = value["y"] / 100
        w = value["width"] / 100
        h = value["height"] / 100

        cx = x + w / 2
        cy = y + h / 2

        yolo_lines.append(
            f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        )

    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))

    print(f"[GT SAVED] {label_path}")

