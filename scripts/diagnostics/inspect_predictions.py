
from pathlib import Path
from ultralytics import YOLO
import cv2

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "parking_space_segmentation_best.pt"
IMAGE_DIR = ROOT / "datasets" / "parking_segmentation" / "images" / "test"

OUTPUT = ROOT / "runs" / "parking_segmentation" / "confidence_diagnostic"
OUTPUT.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL_PATH))

images = sorted(IMAGE_DIR.glob("*.jpg"))[:10]

thresholds = [0.01, 0.05, 0.10, 0.25, 0.50]

for image_path in images:
    image = cv2.imread(str(image_path))

    if image is None:
        continue

    print(f"\nImage: {image_path.name}")

    for threshold in thresholds:
        result = model.predict(
            source=image,
            imgsz=640,
            conf=threshold,
            iou=0.7,
            device=0,
            verbose=False
        )[0]

        count = 0 if result.masks is None else len(result.masks)

        print(f"Confidence {threshold:.2f}: {count} detections")

    # Save a low-confidence prediction for inspection.
    result = model.predict(
        source=image,
        imgsz=640,
        conf=0.01,
        iou=0.7,
        device=0,
        verbose=False
    )[0]

    annotated = result.plot()

    cv2.imwrite(
        str(OUTPUT / image_path.name),
        annotated
    )

print("\nSaved confidence diagnostics:", OUTPUT)
