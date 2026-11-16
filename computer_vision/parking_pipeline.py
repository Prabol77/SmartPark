from pathlib import Path

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from computer_vision.predict_parking import (
    ParkingOccupancyCNN,
    load_checkpoint,
)


class ParkingPipeline:
    """
    YOLO segmentation -> parking-space crops -> CNN occupancy classification.
    """

    def __init__(
        self,
        segmentation_checkpoint,
        occupancy_checkpoint,
        yolo_confidence=0.15,
        yolo_iou=0.50,
    ):
        self.segmentation_checkpoint = Path(segmentation_checkpoint)
        self.occupancy_checkpoint = Path(occupancy_checkpoint)

        if not self.segmentation_checkpoint.is_file():
            raise FileNotFoundError(self.segmentation_checkpoint)

        if not self.occupancy_checkpoint.is_file():
            raise FileNotFoundError(self.occupancy_checkpoint)

        self.yolo_confidence = yolo_confidence
        self.yolo_iou = yolo_iou

        # Load YOLO segmentation model.
        self.yolo = YOLO(str(self.segmentation_checkpoint))

        # Select CNN device.
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load existing occupancy CNN.
        self.cnn = ParkingOccupancyCNN(num_classes=2)

        checkpoint = load_checkpoint(
            self.cnn,
            self.occupancy_checkpoint,
            self.device,
        )

        # Respect the class mapping stored in the checkpoint.
        self.empty_index = 0
        self.occupied_index = 1

        if isinstance(checkpoint, dict):
            class_to_idx = checkpoint.get("class_to_idx")

            if class_to_idx:
                if "Empty" in class_to_idx and "Occupied" in class_to_idx:
                    self.empty_index = int(class_to_idx["Empty"])
                    self.occupied_index = int(class_to_idx["Occupied"])

        if self.empty_index == self.occupied_index:
            raise ValueError("Invalid CNN class mapping.")

        self.preprocess = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.cnn.eval()

        self.yolo_device = (
            0 if torch.cuda.is_available() else "cpu"
        )

    @staticmethod
    def extract_masked_crop(image_bgr, polygon):
        """
        Extract an expanded, masked, square crop from a segmentation polygon.
        """

        image_h, image_w = image_bgr.shape[:2]

        polygon = np.asarray(polygon, dtype=np.float32)

        if polygon.ndim != 2 or polygon.shape[0] < 3:
            return None

        if not np.isfinite(polygon).all():
            return None

        # Expand around the polygon centre, matching the existing
        # crop extraction strategy.
        center = polygon.mean(axis=0)

        expanded = polygon.copy()
        expanded[:, 0] = (
            center[0] + (polygon[:, 0] - center[0]) * 1.15
        )
        expanded[:, 1] = (
            center[1] + (polygon[:, 1] - center[1]) * 1.10
        )

        expanded = np.round(expanded).astype(np.int32)

        expanded[:, 0] = np.clip(expanded[:, 0], 0, image_w - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, image_h - 1)

        x, y, width, height = cv2.boundingRect(
            expanded.reshape(-1, 1, 2)
        )

        if width < 8 or height < 8:
            return None

        crop = image_bgr[
            y:y + height,
            x:x + width
        ].copy()

        local_polygon = expanded - np.array([x, y])

        mask = np.zeros((height, width), dtype=np.uint8)

        cv2.fillPoly(
            mask,
            [local_polygon.reshape(-1, 1, 2)],
            255,
        )

        crop = cv2.bitwise_and(crop, crop, mask=mask)

        # Pad to a square without stretching the parking space.
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
            cv2.BORDER_REPLICATE,
        )

        return square_crop, expanded

    def analyse(self, image_bgr):
        """
        Analyse one BGR image.

        Returns:
            rows: occupancy predictions
            annotated_rgb: visualised detections
            device: CNN inference device
            detection_count: YOLO detections before crop filtering
        """

        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Invalid input image.")

        # Run YOLO segmentation.
        results = self.yolo.predict(
            source=image_bgr,
            imgsz=640,
            conf=self.yolo_confidence,
            iou=self.yolo_iou,
            device=self.yolo_device,
            verbose=False,
        )

        result = results[0]

        if result.masks is None or result.boxes is None:
            raise ValueError(
                "YOLO did not detect any parking-space masks."
            )

        polygons = result.masks.xy
        detection_count = len(polygons)

        rows = []
        annotated = image_bgr.copy()

        # Keep track of previously drawn labels to minimise overlap.
        label_rectangles = []

        image_h, image_w = annotated.shape[:2]

        with torch.inference_mode():

            for index, polygon in enumerate(polygons):

                extracted = self.extract_masked_crop(
                    image_bgr,
                    polygon,
                )

                if extracted is None:
                    continue

                crop_bgr, expanded_polygon = extracted

                crop_rgb = cv2.cvtColor(
                    crop_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                crop_image = Image.fromarray(crop_rgb)

                tensor = (
                    self.preprocess(crop_image)
                    .unsqueeze(0)
                    .to(self.device)
                )

                logits = self.cnn(tensor)

                probabilities = torch.softmax(
                    logits,
                    dim=1,
                )[0].cpu().numpy()

                predicted_index = int(np.argmax(probabilities))

                if predicted_index == self.empty_index:
                    prediction = "EMPTY"

                elif predicted_index == self.occupied_index:
                    prediction = "OCCUPIED"

                else:
                    raise ValueError("Unexpected CNN class index.")

                confidence = float(
                    probabilities[predicted_index]
                ) * 100.0

                if confidence >= 95:
                    confidence_status = "HIGH_CONFIDENCE"

                elif confidence >= 80:
                    confidence_status = "REVIEW"

                else:
                    confidence_status = "UNCERTAIN"

                yolo_confidence = float(
                    result.boxes.conf[index].item()
                )

                slot_id = f"P{index + 1:03d}"

                rows.append({
                    "slot_id": slot_id,
                    "prediction": prediction,
                    "confidence_percent": round(confidence, 2),
                    "confidence_status": confidence_status,
                    "empty_probability": round(
                        float(probabilities[self.empty_index]) * 100,
                        2,
                    ),
                    "occupied_probability": round(
                        float(probabilities[self.occupied_index]) * 100,
                        2,
                    ),
                    "segmentation_confidence": round(
                        yolo_confidence * 100,
                        2,
                    ),
                })

                # ------------------------------------------------
                # CLEAN PARKING-SPACE ANNOTATION
                # ------------------------------------------------

                points = expanded_polygon.reshape(-1, 1, 2)

                if confidence_status == "REVIEW":
                    color = (0, 190, 255)       # Amber

                elif confidence_status == "UNCERTAIN":
                    color = (0, 140, 255)       # Orange

                elif prediction == "EMPTY":
                    color = (40, 190, 70)        # Green

                else:
                    color = (50, 50, 225)        # Red

                # Draw a thinner polygon outline.
                cv2.polylines(
                    annotated,
                    [points],
                    True,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                # Use a compact label.
                label = (
                    f"{slot_id} {prediction} {confidence:.0f}%"
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.32
                thickness = 1
                padding = 4

                (text_w, text_h), baseline = cv2.getTextSize(
                    label,
                    font,
                    font_scale,
                    thickness,
                )

                label_w = text_w + padding * 2
                label_h = text_h + baseline + padding * 2

                # Find polygon boundaries.
                min_x = int(np.min(expanded_polygon[:, 0]))
                min_y = int(np.min(expanded_polygon[:, 1]))
                max_x = int(np.max(expanded_polygon[:, 0]))
                max_y = int(np.max(expanded_polygon[:, 1]))

                # Candidate positions: above, below, left and right.
                candidates = [
                    (min_x, min_y - label_h - 3),
                    (min_x, max_y + 3),
                    (max_x - label_w, min_y - label_h - 3),
                    (max_x - label_w, max_y + 3),
                    (min_x + 3, min_y + 3),
                ]

                valid_candidates = []

                for candidate_x, candidate_y in candidates:

                    candidate_x = max(
                        0,
                        min(candidate_x, image_w - label_w),
                    )

                    candidate_y = max(
                        0,
                        min(candidate_y, image_h - label_h),
                    )

                    rect = (
                        candidate_x,
                        candidate_y,
                        candidate_x + label_w,
                        candidate_y + label_h,
                    )

                    # Calculate overlap with existing labels.
                    overlap_area = 0

                    for old in label_rectangles:
                        ix1 = max(rect[0], old[0])
                        iy1 = max(rect[1], old[1])
                        ix2 = min(rect[2], old[2])
                        iy2 = min(rect[3], old[3])

                        if ix2 > ix1 and iy2 > iy1:
                            overlap_area += (
                                (ix2 - ix1) * (iy2 - iy1)
                            )

                    valid_candidates.append(
                        (overlap_area, candidate_x, candidate_y, rect)
                    )

                    if overlap_area == 0:
                        break

                # Prefer a position that does not overlap another label.
                _, label_x, label_y, label_rect = min(
                    valid_candidates,
                    key=lambda item: item[0],
                )

                label_rectangles.append(label_rect)

                # Draw a solid status-coloured background.
                cv2.rectangle(
                    annotated,
                    (label_x, label_y),
                    (
                        label_x + label_w,
                        label_y + label_h,
                    ),
                    color,
                    -1,
                )

                # Add a subtle dark border.
                cv2.rectangle(
                    annotated,
                    (label_x, label_y),
                    (
                        label_x + label_w,
                        label_y + label_h,
                    ),
                    (25, 25, 25),
                    1,
                )

                # Draw readable white text.
                cv2.putText(
                    annotated,
                    label,
                    (
                        label_x + padding,
                        label_y + padding + text_h,
                    ),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        if not rows:
            raise RuntimeError(
                "YOLO detected masks, but no valid crops were classified."
            )

        annotated_rgb = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB,
        )

        return {
            "rows": rows,
            "annotated_rgb": annotated_rgb,
            "device": str(self.device),
            "detection_count": detection_count,
        }