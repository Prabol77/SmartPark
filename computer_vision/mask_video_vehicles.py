
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# --------------------------------------------------
# Configuration
# --------------------------------------------------

VIDEO_PATH = Path("datasets/public/parking_video.mp4")
OUTPUT_DIR = Path("runs/video_analysis/vehicle_masking")

MODEL_PATH = "yolov8n-seg.pt"
SAMPLE_TIMES = [0, 5, 10, 20, 30, 45, 60, 75, 90, 100]

CONFIDENCE = 0.25
IMAGE_SIZE = 960

# COCO classes:
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Expand bounding boxes to avoid retaining vehicle edges.
PADDING_RATIO = 0.15
EXTRA_DILATION = 9



def detect_and_mask(model, frame):
    height, width = frame.shape[:2]

    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        classes=VEHICLE_CLASSES,
        verbose=False
    )

    result = results[0]

    vehicle_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    annotated = frame.copy()
    detections = []

    boxes = result.boxes
    masks = result.masks

    if boxes is not None and masks is not None:
        for index, box in enumerate(boxes):
            class_id = int(box.cls[0].cpu())

            if class_id not in VEHICLE_CLASSES:
                continue

            confidence = float(box.conf[0].cpu())

            # Get the actual instance segmentation mask.
            instance_mask = (
                masks.data[index].cpu().numpy()
            )

            instance_mask = cv2.resize(
                instance_mask,
                (width, height),
                interpolation=cv2.INTER_LINEAR
            )

            instance_mask = (
                instance_mask > 0.5
            ).astype(np.uint8) * 255

            # Add a small margin around the vehicle silhouette.
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 9)
            )

            instance_mask = cv2.dilate(
                instance_mask,
                kernel,
                iterations=1
            )

            vehicle_mask = cv2.bitwise_or(
                vehicle_mask,
                instance_mask
            )

            x1, y1, x2, y2 = (
                box.xyxy[0].cpu().numpy().astype(int)
            )

            detections.append({
                "class_id": class_id,
                "confidence": round(confidence, 4),
                "box": [x1, y1, x2, y2]
            })

            # Draw the bounding box for reference.
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"Vehicle {confidence:.2f}",
                (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    valid_mask = cv2.bitwise_not(vehicle_mask)

    # Red overlay indicates excluded vehicle pixels.
    overlay = frame.copy()
    overlay[vehicle_mask > 0] = (0, 0, 255)

    mask_preview = cv2.addWeighted(
        frame,
        0.65,
        overlay,
        0.35,
        0
    )

    unmasked = frame.copy()
    unmasked[vehicle_mask > 0] = (110, 110, 110)

    return {
        "annotated": annotated,
        "vehicle_mask": vehicle_mask,
        "valid_mask": valid_mask,
        "unmasked": unmasked,
        "mask_preview": mask_preview,
        "detections": detections
    }

    # Slightly enlarge masks to remove vehicle-edge artifacts.
    kernel_size = EXTRA_DILATION * 2 + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    vehicle_mask = cv2.dilate(
        vehicle_mask,
        kernel,
        iterations=1
    )

    # White = vehicle/excluded.
    # White pixels must NOT be used for parking-line detection.
    valid_mask = cv2.bitwise_not(vehicle_mask)

    # Visual preview: excluded vehicle regions become gray.
    unmasked = frame.copy()
    unmasked[vehicle_mask > 0] = (110, 110, 110)

    # Red overlay shows the excluded areas.
    overlay = frame.copy()
    overlay[vehicle_mask > 0] = (0, 0, 255)

    mask_preview = cv2.addWeighted(
        frame,
        0.65,
        overlay,
        0.35,
        0
    )

    return {
        "annotated": annotated,
        "vehicle_mask": vehicle_mask,
        "valid_mask": valid_mask,
        "unmasked": unmasked,
        "mask_preview": mask_preview,
        "detections": detections
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(VIDEO_PATH)

    print("Loading YOLO model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open parking video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = (
        cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    )

    print(f"Video duration: {duration:.2f}s")
    print(f"Sampling {len(SAMPLE_TIMES)} frames")

    contact_frames = []

    for timestamp in SAMPLE_TIMES:
        if timestamp >= duration:
            continue

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000
        )

        success, frame = cap.read()

        if not success:
            print(f"Could not read frame at {timestamp}s")
            continue

        result = detect_and_mask(model, frame)

        prefix = OUTPUT_DIR / f"frame_{timestamp:03d}"

        cv2.imwrite(
            str(prefix) .__str__() + "_detections.jpg",
            result["annotated"]
        )

        cv2.imwrite(
            str(prefix) .__str__() + "_vehicle_mask.png",
            result["vehicle_mask"]
        )

        cv2.imwrite(
            str(prefix) .__str__() + "_valid_mask.png",
            result["valid_mask"]
        )

        cv2.imwrite(
            str(prefix) .__str__() + "_unmasked.jpg",
            result["unmasked"]
        )

        cv2.imwrite(
            str(prefix) .__str__() + "_mask_preview.jpg",
            result["mask_preview"]
        )

        contact_frames.append(result["mask_preview"])

        print(
            f"{timestamp:3d}s | "
            f"Vehicles detected: {len(result['detections'])}"
        )

    cap.release()

    # Create a contact sheet of the sampled mask previews.
    if contact_frames:
        thumbnails = []

        for frame in contact_frames:
            thumb = cv2.resize(frame, (480, 270))
            thumbnails.append(thumb)

        columns = 2
        rows = (len(thumbnails) + columns - 1) // columns

        sheet = np.zeros(
            (rows * 270, columns * 480, 3),
            dtype=np.uint8
        )

        for index, thumb in enumerate(thumbnails):
            row = index // columns
            col = index % columns

            sheet[
                row * 270:(row + 1) * 270,
                col * 480:(col + 1) * 480
            ] = thumb

        cv2.imwrite(
            str(OUTPUT_DIR / "vehicle_mask_contact_sheet.jpg"),
            sheet
        )

    print("\nVehicle masking completed.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
