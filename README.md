# Autonomous Driving: Vehicle Detection + Tesla Autopilot Safety Analysis

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![YOLOv5](https://img.shields.io/badge/YOLOv5-object%20detection-orange)
![Colab](https://img.shields.io/badge/Google%20Colab-notebook-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

> AIML Capstone Project — two-part study on autonomous driving safety:  
> **(1)** Real-time vehicle detection using YOLOv5, trained on a custom dataset  
> **(2)** Exploratory data analysis of Tesla Autopilot accident records

---

## Project Overview

This capstone explores two complementary angles of autonomous driving:

| Part | Notebook | Technique | Goal |
|---|---|---|---|
| 1 | `02_vehicle_detection_yolov5.ipynb` | YOLOv5 (transfer learning) | Detect and localise vehicles in road images |
| 2 | `01_tesla_autopilot_eda.ipynb` | EDA + statistical analysis | Understand accident patterns in Tesla Autopilot incidents |

---

## Part 1: Vehicle Detection with YOLOv5

### What it does
- Loads a custom vehicle image dataset from Google Drive
- Generates YOLO-format `.txt` label files from a raw CSV annotation file
- Splits data into train/val sets in YOLO directory structure
- Fine-tunes **YOLOv5s** (pretrained on COCO) for 50 epochs
- Runs inference on the validation set and visualises detections

### Pipeline

```
Raw dataset (images + labels.csv)
        │
        ▼
Label preprocessing → filtered_labels.csv → YOLO .txt files
        │
        ▼
Train/Val split (YOLO folder structure)
        │
        ▼
YOLOv5s fine-tuning (50 epochs, img=640, batch=16)
        │
        ▼
best.pt → Inference → Detection visualisations
```

### Key technical choices
- **YOLOv5s** chosen for speed/accuracy balance — suitable for real-time autonomous driving use
- Transfer learning from COCO weights reduces training time and data requirements
- Labels converted from CSV bounding boxes to normalised YOLO format

---

## Part 2: Tesla Autopilot Accident EDA

### Dataset
Public Tesla Autopilot accident records (`Tesla - Deaths.csv`), containing date, location, Tesla model, Autopilot engagement status, deaths, and victim type.

### Analysis performed

| Analysis | Finding |
|---|---|
| Events by year | Accident count rises sharply after 2016 (Autopilot rollout) |
| Events by state | California and Florida account for the majority of incidents |
| Model breakdown | Model S has the highest incident count |
| Victim type | Driver fatalities are the most common outcome |
| Correlation heatmap | Deaths correlate with Autopilot engagement flag |

### Data cleaning steps
- Parsed `Date` column to datetime; dropped rows with unparseable dates
- Stripped whitespace from all column names
- Handled missing values in numerical columns before aggregation

---

## Repository Structure

```
capstone-autonomous-driving/
│
├── notebooks/
│   ├── 01_tesla_autopilot_eda.ipynb       # Part 2: EDA on accident records
│   └── 02_vehicle_detection_yolov5.ipynb  # Part 1: YOLOv5 vehicle detection
│
├── docs/
│   └── capstone_report.pdf                # Full project report
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

### Part 1 — Vehicle Detection (Google Colab)

The detection notebook is designed for **Google Colab** (uses GPU, Google Drive mounting, and installs YOLOv5 automatically).

1. Upload `02_vehicle_detection_yolov5.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Place your dataset in Google Drive under `/MyDrive/data/`
3. Run all cells — YOLOv5 is cloned and installed inside the notebook

> **GPU recommended:** Training 50 epochs on CPU will take several hours. Use Colab's free T4 GPU runtime.

### Part 2 — Tesla EDA (local or Colab)

```bash
# Local
pip install -r requirements.txt
jupyter notebook notebooks/01_tesla_autopilot_eda.ipynb
```

Or upload to Colab directly. The dataset (`Tesla - Deaths.csv`) is not included — download it from [Kaggle: Tesla Deaths](https://www.kaggle.com/datasets) and place it in `/content/` when running on Colab, or update the path in cell 1 for local runs.

---

## Technologies Used

| Technology | Role |
|---|---|
| YOLOv5 (Ultralytics) | Object detection — fine-tuned for vehicle localisation |
| PyTorch | Deep learning backend for YOLOv5 |
| OpenCV | Image processing and label validation |
| Pandas | Data loading, cleaning, aggregation |
| Matplotlib / Seaborn | Visualisation — bar charts, heatmaps, trend lines |
| Google Colab | Cloud GPU training environment |

---

## Key Results

**Part 1 — Vehicle Detection:**
- YOLOv5s successfully detects and localises vehicles in road images after fine-tuning
- Trained for 50 epochs with 640×640 input resolution, batch size 16

**Part 2 — Tesla EDA:**
- Incidents increased ~3× after 2016 (Autopilot v2 launch)
- California accounts for ~30% of all reported incidents
- Tesla Model S is the most frequently involved model
- Autopilot engagement is flagged in the majority of fatal incidents after 2018

---

## Author

**Kishore Kumar Lakshmanan** — AIML Capstone, 2024

---

## License

MIT License — free for research and educational use.
