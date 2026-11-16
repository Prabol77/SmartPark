
from pathlib import Path
from ultralytics import YOLO
import cv2

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "parking_space_segmentation_best.pt"
IMAGE_DIR = ROOT / "datasets" / "parking_segmentation" / "images" / "test"
LABEL_DIR = ROOT / "datasets" / "parking_segmentation" / "labels" / "test"

model = YOLO(str(MODEL_PATH))

images = sorted(IMAGE_DIR.glob("*.jpg"))[:20]

thresholds = [0.01, 0.05, 0.10, 0.25, 0.50]

print("\n========== GROUND TRUTH VS PREDICTIONS ==========\n")

for image_path in images:
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    ground_truth_count = len([
        line for line in label_path.read_text().splitlines()
        if line.strip()
    ])

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    print(f"\n{image_path.name}")
    print(f"Ground truth spaces: {ground_truth_count}")

    for threshold in thresholds:
        result = model.predict(
            source=image,
            imgsz=640,
            conf=threshold,
            iou=0.7,
            device=0,
            verbose=False
        )[0]

        prediction_count = (
            0 if result.masks is None else len(result.masks)
        )

        print(
            f"  conf={threshold:.2f}: "
            f"{prediction_count} predictions"
        )
