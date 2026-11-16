
from pathlib import Path
from collections import Counter

import cv2

DATASET = Path(
    "datasets/public/PKLot/PKLot/PKLotSegmented"
)

classes = Counter()
dimensions = Counter()
extensions = Counter()

for image_path in DATASET.rglob("*"):

    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in {
        ".jpg", ".jpeg", ".png"
    }:
        continue

    label = image_path.parent.name

    classes[label] += 1
    extensions[image_path.suffix.lower()] += 1

    # Inspect only the first 100 images
    if sum(dimensions.values()) < 100:
        image = cv2.imread(str(image_path))

        if image is not None:
            height, width = image.shape[:2]
            dimensions[(width, height)] += 1

print("\n========== CLASS DISTRIBUTION ==========")

for label, count in classes.items():
    print(f"{label}: {count:,}")

print("\n========== IMAGE EXTENSIONS ==========")

for extension, count in extensions.items():
    print(extension, count)

print("\n========== SAMPLE IMAGE DIMENSIONS ==========")

for dimension, count in dimensions.items():
    print(f"{dimension}: {count} images")
