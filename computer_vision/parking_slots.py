"""
SmartPark parking-slot mapper.

Run from the project root:
    python computer_vision/parking_slots.py --image "path/to/parking_lot.jpg"

OpenCV controls:
    Left click : add a polygon vertex
    Enter      : finish the current slot
    U          : undo last vertex (or last completed slot)
    R          : clear current polygon
    S          : save slot map JSON
    Q / Esc    : quit

Extract crops later:
    python computer_vision/parking_slots.py --extract "path/to/slots.json"
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP_DIR = PROJECT_ROOT / "datasets" / "processed" / "parking_slot_maps"
DEFAULT_CROP_DIR = PROJECT_ROOT / "datasets" / "processed" / "parking_slot_crops"


def display_scale(width, height, max_width=1400, max_height=850):
    return min(max_width / width, max_height / height, 1.0)


def draw_overlay(image, slots, current_points, scale):
    canvas = image.copy()

    for slot in slots:
        pts = np.round(np.asarray(slot["points"]) * scale).astype(np.int32)
        cv2.polylines(canvas, [pts], True, (50, 220, 80), 2)
        center = np.mean(pts, axis=0).astype(int)
        cv2.putText(canvas, slot["slot_id"], tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 80), 2,
                    cv2.LINE_AA)

    if current_points:
        pts = np.round(np.asarray(current_points) * scale).astype(np.int32)
        for point in pts:
            cv2.circle(canvas, tuple(point), 5, (0, 200, 255), -1)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], False, (0, 200, 255), 2)

    instructions = [
        "LEFT CLICK: add point | ENTER: finish slot",
        "U: undo | R: clear current | S: save | Q: quit",
        f"Completed slots: {len(slots)} | Current vertices: {len(current_points)}"
    ]
    y = 28
    for line in instructions:
        cv2.putText(canvas, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 27
    return canvas


def save_slot_map(image_path, image, slots, output_path):
    height, width = image.shape[:2]
    payload = {
        "source_image": str(Path(image_path).resolve()),
        "image_width": width,
        "image_height": height,
        "coordinate_format": "pixel_xy_and_normalized_xy",
        "slots": []
    }

    for slot in slots:
        payload["slots"].append({
            "slot_id": slot["slot_id"],
            "points": slot["points"],
            "points_normalized": [
                [round(x / width, 8), round(y / height, 8)]
                for x, y in slot["points"]
            ]
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {len(slots)} parking slots to:\n{output_path}")


def annotate_image(image_path, output_path=None):
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    height, width = image.shape[:2]
    scale = display_scale(width, height)
    output_path = Path(output_path) if output_path else (
        DEFAULT_MAP_DIR / f"{image_path.stem}_slots.json"
    )

    slots = []
    current_points = []
    window_name = "SmartPark - Parking Slot Mapper"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            original_x = max(0, min(width - 1, int(round(x / scale))))
            original_y = max(0, min(height - 1, int(round(y / scale))))
            current_points.append([original_x, original_y])

    cv2.setMouseCallback(window_name, on_mouse)
    print("\nDraw one polygon per parking space.")
    print("Click at least 3 corners, then press ENTER to finish the slot.")

    while True:
        display = draw_overlay(image, slots, current_points, scale)
        display = cv2.resize(
            display,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        )
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):
            if len(current_points) < 3:
                print("A slot needs at least 3 vertices.")
                continue
            slot_id = f"P{len(slots) + 1:03d}"
            slots.append({"slot_id": slot_id,
                          "points": [point[:] for point in current_points]})
            print(f"Added {slot_id} with {len(current_points)} vertices.")
            current_points.clear()

        elif key in (ord("u"), ord("U")):
            if current_points:
                print("Removed vertex:", current_points.pop())
            elif slots:
                print("Removed completed slot:", slots.pop()["slot_id"])

        elif key in (ord("r"), ord("R")):
            current_points.clear()
            print("Current polygon cleared.")

        elif key in (ord("s"), ord("S")):
            if current_points:
                print("Finish or clear the current polygon before saving.")
            elif not slots:
                print("No slots to save yet.")
            else:
                save_slot_map(image_path, image, slots, output_path)

        elif key in (ord("q"), ord("Q"), 27):
            print("Mapper closed.")
            break

    cv2.destroyAllWindows()
    if slots and not output_path.exists():
        print("Mapped slots were not saved. Reopen the mapper and press S before quitting.")


def order_quad(points):
    pts = np.asarray(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(sums)],      # top-left
        pts[np.argmin(diffs)],     # top-right
        pts[np.argmax(sums)],      # bottom-right
        pts[np.argmax(diffs)]      # bottom-left
    ], dtype=np.float32)


def extract_slot_crops(map_path, output_dir=None):
    map_path = Path(map_path).expanduser().resolve()
    if not map_path.exists():
        raise FileNotFoundError(f"Slot map not found: {map_path}")

    data = json.loads(map_path.read_text(encoding="utf-8"))
    image_path = Path(data["source_image"])
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read source image: {image_path}")

    output_dir = Path(output_dir) if output_dir else (
        DEFAULT_CROP_DIR / image_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for slot in data["slots"]:
        points = np.asarray(slot["points"], dtype=np.int32)

        if len(points) == 4:
            quad = order_quad(points)
            tl, tr, br, bl = quad
            crop_width = max(1, int(round(max(
                np.linalg.norm(br - bl), np.linalg.norm(tr - tl)
            ))))
            crop_height = max(1, int(round(max(
                np.linalg.norm(tr - br), np.linalg.norm(tl - bl)
            ))))
            destination = np.array([
                [0, 0], [crop_width - 1, 0],
                [crop_width - 1, crop_height - 1],
                [0, crop_height - 1]
            ], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(quad, destination)
            crop = cv2.warpPerspective(image, matrix, (crop_width, crop_height))
        else:
            x, y, w, h = cv2.boundingRect(points)
            crop = image[y:y + h, x:x + w].copy()
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points - np.array([x, y])], 255)
            crop[mask == 0] = 0

        if crop.size == 0:
            print(f"Skipped empty crop: {slot['slot_id']}")
            continue

        output_path = output_dir / f"{slot['slot_id']}.jpg"
        if cv2.imwrite(str(output_path), crop):
            saved += 1
        else:
            print(f"Could not save: {output_path}")

    print(f"Extracted {saved} slot crops to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Map parking-space polygons on a parking-lot image."
    )
    parser.add_argument("--image", help="Path to parking-lot image.")
    parser.add_argument("--output", help="Optional output JSON path.")
    parser.add_argument("--extract", help="Extract crops using a saved slot-map JSON.")
    parser.add_argument("--crop-output", help="Optional crop output directory.")
    args = parser.parse_args()

    if args.extract:
        extract_slot_crops(args.extract, args.crop_output)
    elif args.image:
        annotate_image(args.image, args.output)
    else:
        parser.error("Provide --image for mapping or --extract for crop extraction.")


if __name__ == "__main__":
    main()
