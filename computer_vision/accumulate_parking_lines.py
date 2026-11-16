import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from mask_video_vehicles import (
    detect_and_mask,
    MODEL_PATH,
    CONFIDENCE,
    IMAGE_SIZE
)

VIDEO_PATH = Path("datasets/public/parking_video.mp4")
OUTPUT_DIR = Path("runs/video_analysis/line_evidence")

SAMPLE_INTERVAL = 2.0

# Only inspect the main parking area.
ROI = (120, 100, 1270, 460)

# Pixel-level evidence thresholds.
MIN_VISIBLE_FRAMES = 3
MIN_LINE_RATIO = 0.20


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    x1, y1, x2, y2 = ROI

    visible_count = np.zeros(
        (height, width), dtype=np.uint16
    )

    bright_count = np.zeros(
        (height, width), dtype=np.uint16
    )

    processed = 0

    print(f"Video duration: {duration:.2f}s")
    print(f"Sampling every {SAMPLE_INTERVAL}s")

    for timestamp in np.arange(
        0, duration, SAMPLE_INTERVAL
    ):
        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000
        )

        success, frame = cap.read()

        if not success:
            continue

        result = detect_and_mask(model, frame)

        valid_mask = result["valid_mask"]

        # Keep only the parking region.
        roi_mask = np.zeros(
            (height, width), dtype=np.uint8
        )
        roi_mask[y1:y2, x1:x2] = 255

        valid_roi = cv2.bitwise_and(
            valid_mask, roi_mask
        )

        # Convert frame to HSV for white paint detection.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        white_pixels = cv2.inRange(
            hsv,
            np.array([0, 0, 180], dtype=np.uint8),
            np.array([180, 75, 255], dtype=np.uint8)
        )

        # Apply vehicle and parking-area masks first.
        candidate = cv2.bitwise_and(
            white_pixels,
            valid_roi
        )

        # Keep only elongated structures in diagonal orientations.
        directional_lines = np.zeros_like(candidate)

        kernel_size = 35
        center = kernel_size // 2

        for angle in range(20, 161, 10):
            kernel = np.zeros(
                (kernel_size, kernel_size),
                dtype=np.uint8
            )

            radians = np.deg2rad(angle)
            length = 15

            dx = int(np.cos(radians) * length)
            dy = int(np.sin(radians) * length)

            cv2.line(
                kernel,
                (center - dx, center - dy),
                (center + dx, center + dy),
                1,
                thickness=2
            )

            opened = cv2.morphologyEx(
                candidate,
                cv2.MORPH_OPEN,
                kernel
            )

            directional_lines = cv2.max(
                directional_lines,
                opened
            )

        candidate = directional_lines

        visible_count += (
            valid_roi > 0
        ).astype(np.uint16)

        bright_count += (
            candidate > 0
        ).astype(np.uint16)

        processed += 1

        print(
            f"Processed {timestamp:6.1f}s | "
            f"vehicles: {len(result['detections'])}"
        )

    cap.release()

    if processed == 0:
        raise RuntimeError("No video frames were processed.")

    # Calculate the proportion of visible frames in which
    # each pixel contained a bright marking.
    ratio = np.divide(
        bright_count.astype(np.float32),
        visible_count.astype(np.float32),
        out=np.zeros((height, width), dtype=np.float32),
        where=visible_count > 0
    )

    # Only accept pixels with sufficient observations.
    evidence = (
        (visible_count >= MIN_VISIBLE_FRAMES) &
        (ratio >= MIN_LINE_RATIO)
    ).astype(np.uint8) * 255

    # Remove isolated pixels while retaining line structures.
    evidence = cv2.morphologyEx(
        evidence,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 3)
        )
    )

    # Save probability/evidence visualization.
    heatmap = np.clip(
        ratio * 255, 0, 255
    ).astype(np.uint8)

    heatmap[visible_count < MIN_VISIBLE_FRAMES] = 0

    colored = cv2.applyColorMap(
        heatmap, cv2.COLORMAP_JET
    )

    # Save outputs.
    cv2.imwrite(
        str(OUTPUT_DIR / "parking_line_evidence.png"),
        evidence
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "parking_line_heatmap.jpg"),
        colored
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "visible_pixel_count.png"),
        np.clip(
            visible_count.astype(np.float32) /
            max(processed, 1) * 255,
            0, 255
        ).astype(np.uint8)
    )

    # Overlay detected evidence on a representative frame.
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, reference = cap.read()
    cap.release()

    if success:
        overlay = reference.copy()
        overlay[evidence > 0] = (0, 255, 0)

        combined = cv2.addWeighted(
            reference, 0.75, overlay, 0.65, 0
        )

        cv2.imwrite(
            str(OUTPUT_DIR / "parking_line_overlay.jpg"),
            combined
        )

    print("\nEvidence accumulation completed.")
    print("Frames processed:", processed)
    print("Evidence:", OUTPUT_DIR / "parking_line_evidence.png")
    print("Overlay:", OUTPUT_DIR / "parking_line_overlay.jpg")


if __name__ == "__main__":
    main()