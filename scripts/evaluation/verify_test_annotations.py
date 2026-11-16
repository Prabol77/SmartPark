
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "datasets/parking_segmentation/images/test"
LABEL_DIR = ROOT / "datasets/parking_segmentation/labels/test"

images = sorted(IMAGE_DIR.glob("*.jpg"))[:20]

for image_path in images:
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None or not label_path.exists():
        continue

    height, width = image.shape[:2]

    valid = 0
    invalid = 0

    for line in label_path.read_text().splitlines():
        values = list(map(float, line.split()))

        if len(values) < 7 or (len(values) - 1) % 2:
            invalid += 1
            continue

        coords = np.array(values[1:]).reshape(-1, 2)

        if (
            np.any(coords < 0)
            or np.any(coords > 1)
            or not np.isfinite(coords).all()
        ):
            invalid += 1
            continue

        valid += 1

    print(
        f"{image_path.name}: "
        f"{width}x{height}, "
        f"valid={valid}, invalid={invalid}"
    )
