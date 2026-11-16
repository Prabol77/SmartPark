
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]

PKLOT_ROOT = ROOT / "datasets/public/PKLot/PKLot/PKLot"
IMAGE_DIR = ROOT / "datasets/parking_segmentation/images/test"
LABEL_DIR = ROOT / "datasets/parking_segmentation/labels/test"

images = sorted(IMAGE_DIR.glob("*.jpg"))[:20]

total_checked = 0
missing_xml = 0
count_mismatches = 0
coordinate_mismatches = 0
max_pixel_error = 0.0

print("\n========== XML TO YOLO AUDIT ==========\n")

for image_path in images:
    # Generated name: UFPR04_<hash>_<original image stem>.jpg
    parts = image_path.stem.split("_", 2)

    if len(parts) != 3:
        print("Cannot decode filename:", image_path.name)
        continue

    location, _, original_stem = parts

    matches = [
        p for p in PKLOT_ROOT.rglob(original_stem + ".xml")
        if location in [part.upper() for part in p.parts]
    ]

    if len(matches) != 1:
        print(f"XML lookup problem: {image_path.name} -> {len(matches)} matches")
        missing_xml += 1
        continue

    xml_path = matches[0]
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))
    if image is None or not label_path.exists():
        print("Image or YOLO label missing:", image_path.name)
        missing_xml += 1
        continue

    height, width = image.shape[:2]

    root = ET.parse(xml_path).getroot()
    xml_polygons = []

    for space in root.findall(".//space"):
        contour = space.find("contour")
        if contour is None:
            continue

        points = []
        for point in contour.findall("point"):
            points.append((
                float(point.attrib["x"]),
                float(point.attrib["y"])
            ))

        if len(points) >= 3:
            xml_polygons.append(points)

    yolo_polygons = []

    for line in label_path.read_text().splitlines():
        values = list(map(float, line.split()))
        coords = np.array(values[1:], dtype=np.float64).reshape(-1, 2)
        coords[:, 0] *= width
        coords[:, 1] *= height
        yolo_polygons.append(coords)

    total_checked += 1

    if len(xml_polygons) != len(yolo_polygons):
        count_mismatches += 1
        print(
            f"COUNT MISMATCH {image_path.name}: "
            f"XML={len(xml_polygons)}, YOLO={len(yolo_polygons)}"
        )
        continue

    image_error = 0.0
    image_mismatch = False

    for xml_points, yolo_points in zip(xml_polygons, yolo_polygons):
        xml_points = np.asarray(xml_points, dtype=np.float64)

        if xml_points.shape != yolo_points.shape:
            image_mismatch = True
            continue

        error = float(np.max(np.abs(xml_points - yolo_points)))
        image_error = max(image_error, error)

        if not np.allclose(xml_points, yolo_points, atol=0.001):
            image_mismatch = True

    max_pixel_error = max(max_pixel_error, image_error)

    if image_mismatch:
        coordinate_mismatches += 1

    print(
        f"{image_path.name}: "
        f"XML={len(xml_polygons)}, YOLO={len(yolo_polygons)}, "
        f"max pixel error={image_error:.6f}"
    )

print("\n========== FINAL AUDIT ==========")
print("Images checked:", total_checked)
print("XML lookup/missing problems:", missing_xml)
print("Polygon count mismatches:", count_mismatches)
print("Coordinate mismatches:", coordinate_mismatches)
print("Maximum coordinate error (pixels):", max_pixel_error)
