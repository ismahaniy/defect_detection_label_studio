# Automated Annotation System using Label Studio + YOLO ML Backend

This project provides an automated image annotation system using:

- Label Studio as the annotation interface
- Label Studio ML Backend for auto-labeling using a YOLO model

Workflow:
1. User uploads images via Label Studio
2. YOLO model auto-generates bounding boxes
3. User reviews, edits, and verifies annotations
4. Verified annotations can be exported in YOLO format
5. Export YOLO annotations

This system is designed for defect detection tasks and supports human-in-the-loop annotation.

---

## Requirements

- Python 3.9+
- venv/Conda 
- YOLOv8 (Ultralytics)

---

## Step 1 — Install & Create Environment

Install dependencies:

```bash
git clone https://github.com/ismahaniy/defect_detection_label_studio.git
```
Create Environment
```bash
python -m venv venv
venv\Scripts\activate
```

```bash
cd defect_detection_label_studio
pip install label-studio
```
---

## Step 2 — Start Label Studio

```bash
label-studio start
```

Open browser:

```
http://127.0.0.1:8080 / http://localhost:8080
```

### Create a Project
1. Sign up / Log in
2. Create a new project
   
---

## Step 3 — Get Label Studio API Key
Ensure to enable Legacy Tokens to get API Key(Label Studio -> Organization -> API Key Settings)
1. Click **Avatar (top-right)**
2. Open **Account & Settings**
3. Copy **API Key/Legacy Token**

Paste the API key into:
- `project_root/ls_export/export_yolo.py`
- `project_root/ls_export/rebuild_yolo_dataset.py`

---

## Step 4 — Start Label Studio ML Backend
 Open new terminal(ensure your venv active)
 ```bash
cd defect_detection_label_studio/label-studio-ml-backend
pip install -r my_ml_backend/requirements-base.txt
pip install -r my_ml_backend/requirements.txt
label-studio-ml start .\my_ml_backend
```

1. You should be able to connect to it in Label Studio project Settings > Model > Add Model and provide with the following URL: http://localhost:9090
2. Labeling Interface and copy this code 
```
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image" model_score_threshold="0.25">
    <Label value="dentado" background="#FFA39E"/>
    <Label value="concave" background="#D4380D"/>
    <Label value="perforation" background="#FFC069"/>
  </RectangleLabels>
</View>

```
Start Label Image
1. Upload images
2. Configure bounding box labeling
3. Start annotating / correcting predictions
---
## Step 4 — Export verified data and Evaluate

New terminal (venv active)
Manual Export
1. Click export button
2. Choose YOLO and images
3. Export
4. Extract folder and copy folder path
5. Run pipeline
 ```bash
cd defect_detection_label_studio/label-studio-ml-backend/project_root
python pipeline/run_pipeline.py
```
6. Choose 2 and paste folder path
7. Evaluate

Auto Export
1. Run pipeline
 ```bash
cd defect_detection_label_studio/label-studio-ml-backend/project_root
python pipeline/run_pipeline.py
```
2. Choose 1, copy folder path of uploaded image and paste folder path
3. Evaluate

## Evaluation & Decision Rules

- Metric: **mAP50**
- Fine-tune triggers:
  - `mAP50 < 0.60`
  - `ΔmAP50 < -0.03`
- Deployment trigger:
  - `mAP50_new > mAP50_old + 0.02`

---

I do not recommend proceeding with fine-tuning because it will take a long time without the GPU, and the training data is not included in the repository.

