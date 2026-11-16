
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_IMAGE = (
    ROOT / "datasets" / "public" / "parking_sample.jpg"
)

DEFAULT_SLOT_MAP = (
    ROOT / "datasets" / "processed" /
    "automatic_slot_maps" /
    "parking_sample_auto_slots.json"
)

DEFAULT_REPORT = (
    ROOT / "runs" / "parking_inference" /
    "confidence" / "parking_sample_confidence_report.csv"
)

DEFAULT_OUTPUT = (
    ROOT / "runs" / "parking_inference" /
    "parking_sample_visual_map.jpg"
)

# BGR colors for OpenCV
COLORS = {
    "EMPTY": (40, 200, 70),       # Green
    "OCCUPIED": (50, 50, 235),    # Red
    "REVIEW": (0, 210, 255),      # Yellow
    "UNCERTAIN": (0, 140, 255)    # Orange
}


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions(report_path):

    predictions = {}

    with open(
        report_path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            slot_id = row["slot_id"].strip()

            prediction = row["prediction"].strip().upper()

            confidence = float(
                row["confidence_percent"]
            )

            status = row["confidence_status"].strip().upper()

            predictions[slot_id] = {
                "prediction": prediction,
                "confidence": confidence,
                "status": status
            }

    return predictions


# ============================================================
# LOAD AUTOMATIC SLOT MAP
# ============================================================

def load_slot_map(slot_map_path):

    with open(
        slot_map_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if "slots" not in data:
        raise ValueError("Slot map does not contain 'slots'.")

    return data


# ============================================================
# DRAW LEGEND
# ============================================================

def draw_legend(image):

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (12, 12),
        (330, 132),
        (20, 25, 35),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.82,
        image,
        0.18,
        0,
        image
    )

    entries = [
        ("EMPTY", COLORS["EMPTY"]),
        ("OCCUPIED", COLORS["OCCUPIED"]),
        ("REVIEW", COLORS["REVIEW"]),
        ("UNCERTAIN", COLORS["UNCERTAIN"])
    ]

    y = 39

    for label, color in entries:

        cv2.rectangle(
            image,
            (25, y - 12),
            (43, y + 2),
            color,
            -1
        )

        cv2.putText(
            image,
            label,
            (53, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        y += 25


# ============================================================
# DRAW PARKING POLYGONS
# ============================================================

def draw_parking_slots(image, slots, predictions):

    overlay = image.copy()

    counts = {
        "EMPTY": 0,
        "OCCUPIED": 0,
        "REVIEW": 0,
        "UNCERTAIN": 0,
        "MISSING": 0
    }

    matched_ids = set()

    for slot in slots:

        slot_id = slot["slot_id"]

        points = np.array(
            slot["points"],
            dtype=np.int32
        )

        if points.shape[0] < 3:
            continue

        prediction_data = predictions.get(slot_id)

        if prediction_data is None:

            color = (150, 150, 150)
            label = "NO DATA"

            counts["MISSING"] += 1

        else:

            prediction = prediction_data["prediction"]
            confidence = prediction_data["confidence"]
            status = prediction_data["status"]

            matched_ids.add(slot_id)

            if status == "UNCERTAIN":

                color = COLORS["UNCERTAIN"]
                counts["UNCERTAIN"] += 1

            elif status == "REVIEW":

                color = COLORS["REVIEW"]
                counts["REVIEW"] += 1

            elif prediction in ("EMPTY", "OCCUPIED"):

                color = COLORS[prediction]
                counts[prediction] += 1

            else:

                color = (150, 150, 150)
                counts["MISSING"] += 1

            label = f"{slot_id} {confidence:.0f}%"

        # Translucent polygon fill
        cv2.fillPoly(
            overlay,
            [points],
            color
        )

    # Blend polygon fills with original image
    image = cv2.addWeighted(
        overlay,
        0.30,
        image,
        0.70,
        0
    )

    # Draw polygon borders and labels
    for slot in slots:

        slot_id = slot["slot_id"]

        points = np.array(
            slot["points"],
            dtype=np.int32
        )

        if points.shape[0] < 3:
            continue

        prediction_data = predictions.get(slot_id)

        if prediction_data is None:

            color = (150, 150, 150)
            label = slot_id

        else:

            prediction = prediction_data["prediction"]
            confidence = prediction_data["confidence"]
            status = prediction_data["status"]

            if status == "UNCERTAIN":
                color = COLORS["UNCERTAIN"]

            elif status == "REVIEW":
                color = COLORS["REVIEW"]

            else:
                color = COLORS[prediction]

            label = f"{slot_id} {confidence:.0f}%"

        cv2.polylines(
            image,
            [points],
            True,
            color,
            2,
            cv2.LINE_AA
        )

        # Polygon center
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))

        # Compact label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.30
        thickness = 1

        (text_width, text_height), baseline = (
            cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )
        )

        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        # Keep label within the image
        text_x = max(
            2,
            min(text_x, image.shape[1] - text_width - 3)
        )

        text_y = max(
            text_height + 3,
            min(text_y, image.shape[0] - baseline - 3)
        )

        cv2.rectangle(
            image,
            (text_x - 2, text_y - text_height - 2),
            (text_x + text_width + 2, text_y + baseline + 2),
            (15, 20, 25),
            -1
        )

        cv2.putText(
            image,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return image, counts, matched_ids


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Visualize automatic parking-slot predictions."
    )

    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE)
    )

    parser.add_argument(
        "--map",
        default=str(DEFAULT_SLOT_MAP)
    )

    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT)
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT)
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    slot_map_path = Path(args.map)
    report_path = Path(args.report)
    output_path = Path(args.output)

    for path in (
        image_path,
        slot_map_path,
        report_path
    ):

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    print("\nLoading parking image...")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Could not read parking image.")

    print("Loading automatic slot map...")

    slot_map = load_slot_map(slot_map_path)

    print("Loading CNN predictions...")

    predictions = load_predictions(report_path)

    slots = slot_map["slots"]

    expected_width = slot_map["image_width"]
    expected_height = slot_map["image_height"]

    if (
        image.shape[1] != expected_width
        or image.shape[0] != expected_height
    ):
        raise ValueError(
            "Image dimensions do not match the slot map."
        )

    print("Drawing parking polygons...")

    visual, counts, matched_ids = draw_parking_slots(
        image,
        slots,
        predictions
    )

    draw_legend(visual)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    success = cv2.imwrite(
        str(output_path),
        visual,
        [cv2.IMWRITE_JPEG_QUALITY, 96]
    )

    if not success:
        raise RuntimeError("Could not save visual map.")

    # ========================================================
    # REPORT
    # ========================================================

    print("\n========== SMARTPARK VISUAL MAP ==========")

    print(f"Detected candidates : {len(slots)}")
    print(f"CNN predictions     : {len(predictions)}")
    print(f"Matched slots       : {len(matched_ids)}")

    print("\nVisual classification:")

    print(f"EMPTY               : {counts['EMPTY']}")
    print(f"OCCUPIED            : {counts['OCCUPIED']}")
    print(f"REVIEW              : {counts['REVIEW']}")
    print(f"UNCERTAIN           : {counts['UNCERTAIN']}")
    print(f"MISSING             : {counts['MISSING']}")

    unmatched = set(predictions.keys()) - matched_ids

    if unmatched:
        print("\nPredictions without a matching polygon:")
        print(", ".join(sorted(unmatched)))

    print(f"\nVisual map saved to:\n{output_path}")


if __name__ == "__main__":
    main()
