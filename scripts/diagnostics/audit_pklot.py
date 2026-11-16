
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
root = ROOT / "datasets" / "public" / "PKLot" / "PKLot" / "PKLot"

xml_files = list(root.rglob("*.xml"))

stats = Counter()
missing_images = []
invalid_polygons = []
total_spaces = 0

for i, xml_path in enumerate(xml_files, 1):
    image_path = xml_path.with_suffix(".jpg")

    if not image_path.exists():
        missing_images.append(str(xml_path))
        continue

    image = cv2.imread(str(image_path))
    if image is None:
        stats["unreadable_images"] += 1
        continue

    h, w = image.shape[:2]
    tree = ET.parse(xml_path)

    for space in tree.getroot().findall("space"):
        total_spaces += 1

        if space.get("occupied") == "1":
            stats["occupied"] += 1
        elif space.get("occupied") == "0":
            stats["empty"] += 1
        else:
            stats["unknown_status"] += 1

        contour = space.find("contour")
        points = contour.findall("point") if contour is not None else []

        if len(points) < 3:
            invalid_polygons.append(str(xml_path))
            continue

        for point in points:
            x = float(point.get("x"))
            y = float(point.get("y"))

            if not (0 <= x < w and 0 <= y < h):
                invalid_polygons.append(str(xml_path))
                break

    if i % 1000 == 0:
        print(f"Audited {i}/{len(xml_files)} XML files")

print("\n========== PKLOT AUDIT ==========")
print("XML files:", len(xml_files))
print("Parking spaces:", total_spaces)
print("Empty:", stats["empty"])
print("Occupied:", stats["occupied"])
print("Unknown status:", stats["unknown_status"])
print("Missing images:", len(missing_images))
print("Unreadable images:", stats["unreadable_images"])
print("Invalid polygons:", len(invalid_polygons))
