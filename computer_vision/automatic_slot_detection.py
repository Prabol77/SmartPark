import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def groups(values, max_gap=4):
    if len(values) == 0:
        return []

    result = []
    start = previous = int(values[0])

    for value in values[1:]:
        value = int(value)
        if value - previous > max_gap:
            result.append((start, previous))
            start = value
        previous = value

    result.append((start, previous))
    return result


def detect_rows(mask):
    h, w = mask.shape
    projection = np.count_nonzero(mask, axis=1).astype(np.float32)
    smooth = cv2.GaussianBlur(
        projection.reshape(-1, 1),
        (1, 0),
        sigmaX=0,
        sigmaY=4
    ).ravel()

    active = np.flatnonzero(smooth >= max(8, w * 0.012))
    bands = []

    for y0, y1 in groups(active, max_gap=max(12, h // 55)):
        if y1 - y0 >= 35:
            bands.append((max(0, y0), min(h - 1, y1)))

    merged = []
    for band in bands:
        if merged and band[0] <= merged[-1][1] + 10:
            merged[-1] = (merged[-1][0], max(merged[-1][1], band[1]))
        else:
            merged.append(band)

    return merged


def detect_dividers(mask, y0, y1):
    band = mask[y0:y1 + 1]
    strength = np.count_nonzero(band, axis=0)
    threshold = max(18, int(band.shape[0] * 0.22))

    active = np.flatnonzero(strength >= threshold)
    candidates = []

    for x0, x1 in groups(active, 3):
        if x1 - x0 <= 10:
            weights = strength[x0:x1 + 1].astype(float)
            positions = np.arange(x0, x1 + 1)
            candidates.append(
                float(np.average(positions, weights=weights))
            )

    merged = []
    for x in candidates:
        if merged and x - merged[-1] < 9:
            merged[-1] = (merged[-1] + x) / 2
        else:
            merged.append(x)

    return merged


def estimate_pitch(xs, min_width, max_width):
    gaps = np.diff(xs)
    valid = gaps[
        (gaps >= min_width * 0.72) &
        (gaps <= max_width * 1.35)
    ]

    return float(np.median(valid)) if len(valid) else None


def reconstruct_dividers(xs, pitch, tolerance):
    if pitch is None or len(xs) < 2:
        return [round(x) for x in xs], []

    result = [xs[0]]
    inferred = []

    for left, right in zip(xs, xs[1:]):
        gap = right - left
        steps = max(1, round(gap / pitch))

        error = abs(gap - steps * pitch) / (steps * pitch)

        if steps > 1 and error <= tolerance:
            for step in range(1, steps):
                x = left + step * pitch
                result.append(x)
                inferred.append(round(x))

        result.append(right)

    return sorted(set(round(x) for x in result)), inferred


def fit_parking_grid(xs, pitch, tolerance=7):
    """
    Fit observed divider positions to a regular parking-grid pattern.
    Returns reconstructed divider coordinates and their evidence.
    """
    if pitch is None or len(xs) < 3:
        return [], []

    xs = np.asarray(xs, dtype=np.float32)

    best_phase = 0
    best_score = -1

    # Search for the grid phase that explains the most observed lines.
    for phase in np.arange(0, pitch, 0.5):
        distances = np.abs(
            xs - (phase + np.round((xs - phase) / pitch) * pitch)
        )
        score = np.sum(distances <= tolerance)

        if score > best_score:
            best_score = score
            best_phase = phase

    indices = np.round((xs - best_phase) / pitch).astype(int)

    supported = {}
    for index, x in zip(indices, xs):
        expected = best_phase + index * pitch

        if abs(x - expected) <= tolerance:
            supported[int(index)] = float(x)

    if len(supported) < 3:
        return [], []

    first = min(supported)
    last = max(supported)

    reconstructed = []
    evidence = []

    for index in range(first, last + 1):
        x = int(round(best_phase + index * pitch))

        reconstructed.append(x)
        evidence.append({
            "x": x,
            "observed": index in supported,
            "inferred": index not in supported
        })

    return reconstructed, evidence


def detect_slots(image, min_width=25, max_width=48,
                 tolerance=0.28):

    h, w = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    bright = cv2.inRange(
        hsv,
        np.array([0, 0, 150], dtype=np.uint8),
        np.array([180, 105, 255], dtype=np.uint8)
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(25, h // 24))
    )

    mask = cv2.morphologyEx(
        bright, cv2.MORPH_OPEN, kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, 11)
        )
    )

    detected_bands = detect_rows(mask)

    slots = []
    reports = []

    for band_number, (band_y0, band_y1) in enumerate(
        detected_bands, 1
    ):

        # Ignore incomplete parking sections at image boundaries.
        if band_y0 < int(h * 0.08) or band_y1 > int(h * 0.90):
            print(f"Skipping incomplete section {band_number}")
            continue

        # Find the vertical parking dividers across the full section.
        raw = detect_dividers(mask, band_y0, band_y1)

        pitch = estimate_pitch(
            raw, min_width, max_width
        )

        grid, evidence = fit_parking_grid(
            raw, pitch, tolerance=7
        )

        if len(grid) < 4:
            print(f"Insufficient grid evidence in section {band_number}")
            continue

        # Each detected section contains two opposing parking rows.
        midpoint = (band_y0 + band_y1) // 2

        individual_rows = [
            (band_y0, midpoint),
            (midpoint, band_y1)
        ]

        for half_index, (y0, y1) in enumerate(
            individual_rows, 1
        ):

            row_number = (band_number - 1) * 2 + half_index
            row_count = 0

            if y1 - y0 < 25:
                continue

            for i in range(len(grid) - 1):

                left = grid[i]
                right = grid[i + 1]

                bay_width = right - left

                if not min_width <= bay_width <= max_width:
                    continue

                if left < 2 or right > w - 3:
                    continue

                left_evidence = evidence[i]
                right_evidence = evidence[i + 1]

                # Avoid creating a bay bounded by two inferred lines.
                if (
                    left_evidence["inferred"]
                    and right_evidence["inferred"]
                ):
                    continue

                x0 = int(left + 1)
                x1 = int(right - 1)

                points = [
                    [x0, y0],
                    [x1, y0],
                    [x1, y1],
                    [x0, y1]
                ]

                normalized = [
                    [round(x / w, 8), round(y / h, 8)]
                    for x, y in points
                ]

                slots.append({
                    "slot_id": f"P{len(slots) + 1:03d}",
                    "row": row_number,
                    "section": band_number,
                    "points": points,
                    "points_normalized": normalized,
                    "left_divider_inferred": left_evidence["inferred"],
                    "right_divider_inferred": right_evidence["inferred"]
                })

                row_count += 1

            reports.append({
                "row": row_number,
                "section": band_number,
                "half": "upper" if half_index == 1 else "lower",
                "y_range": [y0, y1],
                "raw_dividers": len(raw),
                "grid_dividers": len(grid),
                "estimated_bay_pitch_px": (
                    round(pitch, 2) if pitch else None
                ),
                "candidate_count": row_count
            })

    return {
        "image_width": w,
        "image_height": h,
        "coordinate_format": "pixel_xy_and_normalized_xy",
        "detection_method": "split_opposing_rows_with_validated_parking_grid",
        "row_reports": reports,
        "slots": slots
    }


def extract_crops(image, slots, output_dir):
    """
    Extract expanded parking-slot crops.

    - Expands each slot around its center.
    - Preserves additional context around vehicles.
    - Clips coordinates safely to image boundaries.
    - Masks pixels outside the expanded polygon.
    - Pads rectangular crops to square without stretching.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    image_h, image_w = image.shape[:2]
    saved = 0

    # Expansion factors: modest horizontal and vertical margins.
    expand_x = 1.15
    expand_y = 1.10

    for slot in slots:

        original = np.array(
            slot["points"], dtype=np.float32
        )

        # Expand polygon around its center.
        center = np.mean(original, axis=0)

        expanded = original.copy()
        expanded[:, 0] = (
            center[0] +
            (original[:, 0] - center[0]) * expand_x
        )
        expanded[:, 1] = (
            center[1] +
            (original[:, 1] - center[1]) * expand_y
        )

        expanded = np.round(expanded).astype(np.int32)

        # Clip polygon to the source image.
        expanded[:, 0] = np.clip(
            expanded[:, 0], 0, image_w - 1
        )
        expanded[:, 1] = np.clip(
            expanded[:, 1], 0, image_h - 1
        )

        # Calculate crop bounds.
        x, y, width, height = cv2.boundingRect(
            expanded.reshape(-1, 1, 2)
        )

        if width < 8 or height < 8:
            print(f"Skipping invalid crop: {slot['slot_id']}")
            continue

        crop = image[
            y:y + height,
            x:x + width
        ].copy()

        # Create a mask for the expanded parking-space polygon.
        local_polygon = expanded - np.array([x, y])

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        cv2.fillPoly(
            mask,
            [local_polygon.reshape(-1, 1, 2)],
            255
        )

        # Preserve pixels inside the expanded polygon.
        crop = cv2.bitwise_and(
            crop,
            crop,
            mask=mask
        )

        # Convert rectangular crop to a square without distortion.
        side = max(height, width)

        top = (side - height) // 2
        bottom = side - height - top

        left = (side - width) // 2
        right = side - width - left

        square_crop = cv2.copyMakeBorder(
            crop,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_REPLICATE
        )

        # Save high-quality JPEG crop.
        crop_path = output_dir / f"{slot['slot_id']}.jpg"

        success = cv2.imwrite(
            str(crop_path),
            square_crop,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        if success:
            saved += 1
        else:
            print(f"Failed to save: {slot['slot_id']}")

    print(f"Extracted expanded square crops: {saved}")

    return saved


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--extract-crops", action="store_true"
    )
    parser.add_argument(
        "--min-width", type=int, default=25
    )
    parser.add_argument(
        "--max-width", type=int, default=48
    )
    parser.add_argument(
        "--pitch-tolerance", type=float, default=0.28
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    result = detect_slots(
        image,
        args.min_width,
        args.max_width,
        args.pitch_tolerance
    )

    result["source_image"] = str(image_path.resolve())

    output = Path("datasets/processed/automatic_slot_maps")
    output.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    json_path = output / f"{stem}_auto_slots.json"
    preview_path = output / f"{stem}_auto_preview.jpg"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    preview = image.copy()

    for slot in result["slots"]:
        pts = np.array(
            slot["points"], dtype=np.int32
        ).reshape(-1, 1, 2)

        cv2.polylines(
            preview, [pts], True, (0, 220, 255), 2
        )

        center = np.mean(
            pts.reshape(-1, 2), axis=0
        ).astype(int)

        cv2.putText(
            preview,
            slot["slot_id"],
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            2
        )

    cv2.imwrite(str(preview_path), preview)

    print(f"Detected parking rows: {len(result['row_reports'])}")

    for row in result["row_reports"]:
        print(
            f"Row {row['row']} (Section {row['section']} - {row['half']}): "
            f"raw dividers={row['raw_dividers']}, "
            f"grid dividers={row['grid_dividers']}, "
            f"slots={row['candidate_count']}, "
            f"pitch={row['estimated_bay_pitch_px']}"
        )

    print(f"Total candidate slots: {len(result['slots'])}")
    print(f"JSON: {json_path.resolve()}")
    print(f"Preview: {preview_path.resolve()}")

    if args.extract_crops:
        crop_dir = (
            Path("datasets/processed/parking_slot_crops")
            / f"{stem}_auto"
        )

        extract_crops(image, result["slots"], crop_dir)
        print(f"Crops saved: {crop_dir.resolve()}")


if __name__ == "__main__":
    main()