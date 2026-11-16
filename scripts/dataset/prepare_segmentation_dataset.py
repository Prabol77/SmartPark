
import csv
import hashlib
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKLOT_ROOT = PROJECT_ROOT / "datasets" / "public" / "PKLot" / "PKLot" / "PKLot"

OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "parking_segmentation"

LOCATIONS = {"PUCPR", "UFPR04", "UFPR05"}
TEST_LOCATION = "UFPR04"
VALIDATION_RATIO = 0.20
RANDOM_SEED = 42

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def find_location(xml_path):
    for part in xml_path.parts:
        name = part.upper()
        if name in LOCATIONS:
            return name

    return None


def find_image(xml_path):
    for extension in IMAGE_EXTENSIONS:
        candidate = xml_path.with_suffix(extension)
        if candidate.exists():
            return candidate

    return None


def get_date_group(xml_path, location):
    match = re.search(r"\d{4}-\d{2}-\d{2}", xml_path.stem)

    if match:
        return f"{location}_{match.group(0)}"

    # Fallback: keep this file in its own group.
    return f"{location}_{xml_path.stem}"


def polygon_is_valid(points, width, height):
    if len(points) < 3:
        return False

    # Remove duplicate consecutive points.
    cleaned = []

    for point in points:
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)

    if len(set(cleaned)) < 3:
        return False

    for x, y in cleaned:
        if not (0 <= x < width and 0 <= y < height):
            return False

    contour = __import__("numpy").array(
        cleaned, dtype="float32"
    ).reshape((-1, 1, 2))

    if abs(cv2.contourArea(contour)) < 2:
        return False

    return True


def extract_polygons(xml_path, width, height):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    polygons = []
    invalid_count = 0

    for space in root.findall(".//space"):
        contour = space.find("contour")

        if contour is None:
            invalid_count += 1
            continue

        points = []

        for point in contour.findall("point"):
            try:
                x = float(point.attrib["x"])
                y = float(point.attrib["y"])
                points.append((x, y))
            except (KeyError, ValueError):
                points = []
                break

        if not polygon_is_valid(points, width, height):
            invalid_count += 1
            continue

        # YOLO segmentation format:
        # class_id x1 y1 x2 y2 ...
        normalized = []

        for x, y in points:
            normalized.extend([
                min(max(x / width, 0.0), 1.0),
                min(max(y / height, 0.0), 1.0),
            ])

        polygons.append(normalized)

    return polygons, invalid_count


def make_output_stem(xml_path, location):
    relative_path = str(xml_path.relative_to(PKLOT_ROOT))
    short_hash = hashlib.sha1(
        relative_path.encode("utf-8")
    ).hexdigest()[:10]

    return f"{location}_{short_hash}_{xml_path.stem}"


# --------------------------------------------------
# DISCOVER AND GROUP FILES
# --------------------------------------------------

if not PKLOT_ROOT.exists():
    raise FileNotFoundError(
        f"PKLot directory not found: {PKLOT_ROOT}"
    )

xml_files = sorted(PKLOT_ROOT.rglob("*.xml"))

print(f"Found XML files: {len(xml_files)}")

records = []
groups_by_location = defaultdict(set)

for index, xml_path in enumerate(xml_files, start=1):
    location = find_location(xml_path)

    if location is None:
        continue

    image_path = find_image(xml_path)

    if image_path is None:
        continue

    group = get_date_group(xml_path, location)

    records.append({
        "xml": xml_path,
        "image": image_path,
        "location": location,
        "group": group,
    })

    groups_by_location[location].add(group)

    if index % 1000 == 0:
        print(f"Discovered {index}/{len(xml_files)} XML files")


# --------------------------------------------------
# DATE-BASED TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

random.seed(RANDOM_SEED)
validation_groups = set()

for location in ("PUCPR", "UFPR05"):
    groups = sorted(groups_by_location[location])
    random.shuffle(groups)

    if len(groups) >= 2:
        validation_count = max(
            1, round(len(groups) * VALIDATION_RATIO)
        )
        validation_count = min(validation_count, len(groups) - 1)

        validation_groups.update(groups[:validation_count])

print("\nDate groups:")
for location in sorted(groups_by_location):
    print(
        f"{location}: {len(groups_by_location[location])} groups"
    )

print(f"Validation groups: {len(validation_groups)}")


# --------------------------------------------------
# CREATE YOLO DIRECTORY STRUCTURE
# --------------------------------------------------

for split in ("train", "val", "test"):
    (OUTPUT_ROOT / "images" / split).mkdir(
        parents=True, exist_ok=True
    )
    (OUTPUT_ROOT / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )

# --------------------------------------------------
# CONVERT XML POLYGONS TO YOLO LABELS
# --------------------------------------------------

stats = Counter()
split_groups = defaultdict(set)
summary_path = OUTPUT_ROOT / "conversion_summary.csv"

with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow([
        "image",
        "location",
        "split",
        "valid_polygons",
        "invalid_polygons",
    ])

    for index, record in enumerate(records, start=1):
        xml_path = record["xml"]
        image_path = record["image"]
        location = record["location"]
        group = record["group"]

        if location == TEST_LOCATION:
            split = "test"
        elif group in validation_groups:
            split = "val"
        else:
            split = "train"

        image = cv2.imread(str(image_path))

        if image is None:
            stats["unreadable"] += 1
            continue

        height, width = image.shape[:2]

        try:
            polygons, invalid_count = extract_polygons(
                xml_path, width, height
            )
        except Exception as error:
            print(f"XML error: {xml_path} -> {error}")
            stats["xml_errors"] += 1
            continue

        if not polygons:
            stats["images_without_valid_polygons"] += 1
            stats["invalid_polygons"] += invalid_count
            continue

        output_stem = make_output_stem(xml_path, location)

        image_output = (
            OUTPUT_ROOT / "images" / split / f"{output_stem}.jpg"
        )
        label_output = (
            OUTPUT_ROOT / "labels" / split / f"{output_stem}.txt"
        )

        # Store a consistent JPEG copy.
        if not cv2.imwrite(str(image_output), image):
            stats["write_errors"] += 1
            continue

        with label_output.open("w", encoding="utf-8") as label_file:
            for polygon in polygons:
                values = " ".join(f"{value:.6f}" for value in polygon)
                label_file.write(f"0 {values}\n")

        stats[f"{split}_images"] += 1
        stats[f"{split}_polygons"] += len(polygons)
        stats["invalid_polygons"] += invalid_count
        split_groups[split].add(group)

        writer.writerow([
            image_output.name,
            location,
            split,
            len(polygons),
            invalid_count,
        ])

        if index % 500 == 0:
            print(f"Converted {index}/{len(records)} files")


# --------------------------------------------------
# CREATE YOLO DATA CONFIG
# --------------------------------------------------

yaml_path = OUTPUT_ROOT / "data.yaml"

yaml_path.write_text(
    f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: parking_space
""",
    encoding="utf-8",
)

# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("\n========== CONVERSION COMPLETE ==========")
print(f"Output directory: {OUTPUT_ROOT}")
print(f"Train images: {stats['train_images']}")
print(f"Validation images: {stats['val_images']}")
print(f"Test images: {stats['test_images']}")
print(f"Train polygons: {stats['train_polygons']}")
print(f"Validation polygons: {stats['val_polygons']}")
print(f"Test polygons: {stats['test_polygons']}")
print(f"Invalid polygons skipped: {stats['invalid_polygons']}")
print(f"Images without valid polygons: {stats['images_without_valid_polygons']}")
print(f"Unreadable images: {stats['unreadable']}")
print(f"XML errors: {stats['xml_errors']}")
print(f"Write errors: {stats['write_errors']}")
print(f"YOLO config: {yaml_path}")
print(f"Conversion report: {summary_path}")
