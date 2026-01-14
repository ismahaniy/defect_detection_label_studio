import requests
import zipfile
import io
import os
from datetime import datetime

# =========================
# CONFIGURATION
# =========================
LABEL_STUDIO_URL = "http://localhost:8080"
API_KEY = "62bfaba676e48bfe68449aa9580d3bf12e4b3c7a"
PROJECT_ID = 1   # change this
EXPORT_ROOT = "exports"

# =========================
# EXPORT FUNCTION
# =========================
def export_yolo_dataset():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(EXPORT_ROOT, f"yolo_export_{timestamp}")
    os.makedirs(export_dir, exist_ok=True)

    export_url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/export"
    params = {
        "exportType": "YOLO"
    }

    headers = {
        "Authorization": f"Token {API_KEY}"
    }

    print(f"DEBUG: Using URL: {export_url}")
    print(f"DEBUG: Headers: {headers}")

    print("[INFO] Requesting YOLO export from Label Studio...")
    response = requests.get(export_url, headers=headers, params=params)

    if response.status_code != 200:
        raise RuntimeError(
            f"Export failed: {response.status_code} {response.text}"
        )

    print("[INFO] Downloading export ZIP...")
    zip_bytes = io.BytesIO(response.content)

    with zipfile.ZipFile(zip_bytes) as z:
        z.extractall(export_dir)

    print(f"[SUCCESS] YOLO dataset exported to: {export_dir}")
    return export_dir


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    export_yolo_dataset()
