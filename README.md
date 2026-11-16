# SmartPark — Intelligent Parking Management System

SmartPark is an AI-powered parking management prototype that detects parking spaces, classifies their occupancy, and presents parking availability through an interactive Streamlit dashboard.

The project combines computer vision, deep learning, and visual analytics to support more informed parking-space management.

## Features

- **Parking-space detection:** Uses a YOLO-based segmentation model to identify parking spaces in an uploaded parking image.
- **Occupancy classification:** Uses a trained CNN classifier to predict whether each detected space is empty or occupied.
- **Interactive dashboard:** Displays annotated images, occupancy statistics, confidence indicators, and visual summaries.
- **Confidence-based review:** Highlights uncertain predictions for manual inspection.
- **Export:** Provides CSV reports of parking-space predictions.
- **Domain adaptation:** Includes a fine-tuned segmentation model evaluated on a held-out UFPR04 dataset.

## Technology Stack

- Python
- PyTorch and Torchvision
- Ultralytics YOLO
- OpenCV
- Streamlit
- Plotly
- NumPy
- Pillow
- Scikit-learn

## System Workflow

1. Upload a parking-area image through the dashboard.
2. The segmentation model detects individual parking spaces.
3. Detected parking-space regions are extracted.
4. The occupancy classifier predicts `EMPTY` or `OCCUPIED`.
5. The dashboard displays annotated detections, confidence information, and occupancy statistics.
6. Results can be exported for further analysis.

## Project Structure

```text
SmartPark/
├── app.py
├── requirements.txt
├── .gitignore
├── computer_vision/
│   ├── parking_pipeline.py
│   ├── occupancy_classifier.py
│   ├── parking_slots.py
│   ├── vehicle_detection.py
│   └── ...
├── evaluation/
│   ├── add_confidence_status.py
│   └── compare_crop_predictions.py
├── scripts/
│   ├── dataset/
│   ├── diagnostics/
│   ├── evaluation/
│   ├── tests/
│   └── training/
└── datasets/
    └── parking_sample.jpg
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Prabol77/SmartPark.git
cd SmartPark
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Model Checkpoints

Trained model checkpoints and large datasets are intentionally excluded from this repository to keep it lightweight.

The dashboard expects these model files:

| Model | Expected path |
|---|---|
| Fine-tuned YOLO segmentation model | `runs/parking_segmentation_adaptation/ufpr04_finetune_v1/weights/best.pt` |
| Occupancy classification model | `models/best_occupancy_ufpr04_holdout.pth` |

Obtain the corresponding trained checkpoints and place them at these paths before running the application. The repository does not currently include an automated checkpoint download process.

## Run the Application

From the project root:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal by Streamlit.

## Evaluation

The project includes experiments using the PKLot parking dataset, including domain adaptation and held-out UFPR04 evaluation.

Reported model results must be interpreted within their evaluation scope:

- The fine-tuned YOLO model was evaluated on 462 held-out UFPR04 test images.
- The occupancy classifier achieved 98.22% accuracy on held-out UFPR04 segmented parking-space crops.
- These results describe performance on the evaluated dataset and do not establish equivalent performance on arbitrary real-world camera footage.
- Dashboard confidence scores represent model prediction confidence, not independently verified classification accuracy.

## Current Scope and Limitations

- The current dashboard workflow analyses uploaded images.
- Predictions depend on the quality and domain of the input image.
- Low-confidence predictions may require manual review.
- Historical occupancy prediction and live camera integration are not presented as completed dashboard features.

## Contributors

Developed as an academic project for CSE472.

## Repository

[GitHub — SmartPark](https://github.com/Prabol77/SmartPark)