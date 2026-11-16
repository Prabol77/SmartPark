
from pathlib import Path
from collections import Counter

import cv2
import torch
from ultralytics import YOLO


VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


class VehicleDetector:

    def __init__(self, model_path="yolo11n.pt", confidence=0.15):

        self.model = YOLO(model_path)
        self.confidence = confidence

        # Select GPU based on PyTorch CUDA availability
        self.device = 0 if torch.cuda.is_available() else "cpu"

        print("Selected device:", self.device)

        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))

    def detect(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence,
            classes=list(VEHICLE_CLASSES.keys()),
            device=self.device,
            imgsz=960,
            verbose=False
        )

        result = results[0]

        detected_classes = []
        detection_details = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                detected_classes.append(class_id)

                detection_details.append({
                    "vehicle": VEHICLE_CLASSES[class_id],
                    "confidence": round(confidence, 3)
                })

        counts = Counter(
            VEHICLE_CLASSES[class_id]
            for class_id in detected_classes
        )

        annotated_image = result.plot()

        output_dir = Path("runs/vehicle_detection")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{image_path.stem}_detected.jpg"

        cv2.imwrite(str(output_path), annotated_image)

        return {
            "total_vehicles": len(detected_classes),
            "vehicle_counts": dict(counts),
            "detections": detection_details,
            "device": str(self.device),
            "output_image": str(output_path)
        }


if __name__ == "__main__":

    detector = VehicleDetector(
        confidence=0.15
    )

    IMAGE_PATH = "datasets/public/parking_sample.jpg"

    output = detector.detect(IMAGE_PATH)

    print("\n========== DETECTION RESULTS ==========")
    print("Total vehicles:", output["total_vehicles"])
    print("Vehicle counts:", output["vehicle_counts"])
    print("Device:", output["device"])

    print("\nIndividual detections:")
    for detection in output["detections"]:
        print(detection)

    print("\nOutput image:", output["output_image"])
