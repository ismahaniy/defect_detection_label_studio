# Automated Human-in-the-Loop Continual Learning System
YOLOv8 + Label Studio ML Backend

This repository implements a **performance-aware continual learning pipeline** for object/defect detection using **YOLOv8** and **Label Studio**.

The system:
- Uses **human-verified annotations** as ground truth
- Evaluates models using **mAP50**
- Triggers **incremental fine-tuning only when performance degrades**
- Automatically **compares and deploys improved models**
- Prevents **evaluation leakage** and **catastrophic forgetting**

---

## System Workflow

```
Label Studio (Human Verification)
        ↓
Export YOLO Annotations (Snapshot)
        ↓
Rebuild YOLO Dataset (Images + Labels)
        ↓
Evaluate Model (mAP50)
        ↓
Decision Logic
        ↓
(Optional) Incremental Fine-tuning
        ↓
Re-evaluate & Compare Models
        ↓
Deploy Best Model to Label Studio
```

---

## Project Structure

```
project_root/
├── pipeline/
│   └── run_pipeline.py
│
├── ls_export/
│   ├── export_yolo.py
│   ├── rebuild_yolo_dataset.py
│   └── create_yaml.py
│
├── finetune/
│   └── build_fine_tune_dataset.py
│
├── models/
│   ├── deployed/
│   │   └── best.pt
│   ├── candidates/
│   │   └── best_ft.pt
│   └── archive/
│
├── original_training_data/
│   ├── images/
│   └── labels/
│
├── exports/
│   └── yolo_export_xxx/
│
├── history/
│   ├── eval_history.csv
│   └── last_eval_export.txt
│
└── README.md
```

---

## Requirements

- Python 3.9+
- Conda or virtualenv
- YOLOv8 (Ultralytics)
- Label Studio
- Label Studio ML Backend

---

## Step 1 — Create Environment

```bash
conda create -n detection_env python=3.9
conda activate detection_env
```

Install dependencies:

```bash
git clone https://github.com/ismahaniy/defect_detection_label_studio.git
cd defect_detection_label_studio/label-studio-ml-backend
pip install -e .
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
- `ls_export/export_yolo.py`
- `ls_export/rebuild_yolo_dataset.py`

---

## Step 4 — Start Label Studio ML Backend
 Open new terminal
 ```bash
cd defect_detection_label_studio/label-studio-ml-backend
pip install -r my_ml_backend
label-studio-ml start .\my_ml_backend
```

1. You should be able to connect to it in Label Studio project Settings > Machine Learning > Add Model and provide with the following URL: http://localhost:9090
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

Ensure your `model.py` loads only the deployed model:

```python
self.model = YOLO("models/deployed/best.pt")
```

1. Upload images
2. Configure bounding box labeling
3. Start annotating / correcting predictions
---

## Step 5 — Initial Model Setup

Place your initial YOLO model here:

```
models/deployed/best.pt
```

This model is used for:
- Auto-labeling
- Baseline evaluation

---

## Step 6 — First Pipeline Run (Baseline)

```bash
python pipeline/run_pipeline.py
```

What happens:
- Exports human-verified annotations
- Rebuilds YOLO dataset
- Evaluates mAP50
- Saves baseline performance
- **No fine-tuning occurs**

This establishes the reference performance.

---

## Step 7 — Continual Learning Runs

After **more human annotations** are added in Label Studio:

```bash
python pipeline/run_pipeline.py
```

The pipeline will:
1. Export updated annotations
2. Evaluate current deployed model
3. Detect performance degradation
4. Fine-tune using:
   - Previously verified data
   - Historical training data (replay)
5. Compare fine-tuned model vs deployed model
6. Deploy only if performance improves

---

## Evaluation & Decision Rules

- Metric: **mAP50**
- Retraining triggers:
  - `mAP50 < 0.60`
  - `ΔmAP50 < -0.03`
- Deployment trigger:
  - `mAP50_new > mAP50_old + 0.02`

---

## Key Design Principles

### Ground Truth
- Exported Label Studio data acts as **human-verified ground truth**
- Used **only for evaluation**

### Temporal Separation
- Evaluation data is **never used for training in the same cycle**
- Training uses **previously evaluated data only**

### Fine-Tuning Strategy
- Incremental (not from scratch)
- Low learning rate
- Few epochs
- 70% verified data + 30% replay data

---

## Model Deployment

When a fine-tuned model outperforms the deployed model:
- It is automatically promoted to:
  ```
  models/deployed/best.pt
  ```
- Restart ML backend:

```bash
label-studio-ml start my_ml_backend
```

Label Studio will now auto-label using the improved model.

---
