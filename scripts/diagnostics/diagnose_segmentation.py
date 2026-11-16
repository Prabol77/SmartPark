
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

MODEL = ROOT / "models" / "parking_space_segmentation_best.pt"
DATASET = ROOT / "datasets" / "parking_segmentation"

IMAGE_DIR = DATASET / "images" / "test"
LABEL_DIR = DATASET / "labels" / "test"

OUTPUT = ROOT / "runs" / "parking_segmentation" / "diagnostics"
OUTPUT.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL))

images = sorted(IMAGE_DIR.glob("*.jpg"))[:12]

if not images:
    raise RuntimeError("No test images found")

for index, image_path in enumerate(images, 1):
    image = cv2.imread(str(image_path))

    if image is None:
        continue

    height, width = image.shape[:2]
    ground_truth = image.copy()

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if label_path.exists():
        for line in label_path.read_text().splitlines():
            values = list(map(float, line.split()))

            if len(values) < 7:
                continue

            coordinates = np.array(
                values[1:], dtype=np.float32
            ).reshape(-1, 2)

            coordinates[:, 0] *= width
            coordinates[:, 1] *= height

            polygon = np.round(coordinates).astype(np.int32)

            cv2.polylines(
                ground_truth,
                [polygon],
                True,
                (0, 255, 0),
                2
            )

    result = model.predict(
        source=image,
        imgsz=640,
        conf=0.25,
        device=0,
        verbose=False
    )[0]

    predicted = image.copy()

    if result.masks is not None:
        for polygon in result.masks.xy:
            points = np.round(polygon).astype(np.int32)

            cv2.polylines(
                predicted,
                [points],
                True,
                (0, 0, 255),
                2
            )

    comparison = np.hstack([
        ground_truth,
        predicted
    ])

    cv2.putText(
        comparison,
        "GROUND TRUTH - GREEN",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        comparison,
        "YOLO PREDICTIONS - RED",
        (width + 15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    output_path = OUTPUT / f"comparison_{index:02d}.jpg"

    cv2.imwrite(str(output_path), comparison)

    print(
        f"{index}/12 | {image_path.name} | "
        f"Predicted spaces: "
        f"{0 if result.masks is None else len(result.masks)}"
    )

print("\nDiagnostics saved to:", OUTPUT)
