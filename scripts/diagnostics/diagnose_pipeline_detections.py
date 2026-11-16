
from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "runs" / "parking_segmentation_adaptation"
    / "ufpr04_finetune_v1" / "weights" / "best.pt"
)

IMAGE_PATH = ROOT / "datasets" / "public" / "parking_sample.jpg"

OUTPUT = ROOT / "runs" / "detection_diagnostics"


def main():
    model = YOLO(str(MODEL_PATH))

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(IMAGE_PATH)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    for threshold in [0.01, 0.05, 0.10, 0.15, 0.25, 0.40]:
        result = model.predict(
            source=image,
            imgsz=640,
            conf=threshold,
            iou=0.50,
            device=0,
            retina_masks=True,
            verbose=False,
        )[0]

        count = len(result.boxes) if result.boxes is not None else 0

        print(f"\nConfidence threshold: {threshold}")
        print("Detections:", count)

        if count:
            confidences = result.boxes.conf.cpu().tolist()
            print("Confidence scores:", [
                round(x, 3) for x in confidences
            ])

            annotated = result.plot(
                labels=False,
                conf=False,
                boxes=True,
                masks=True,
            )

            output_path = OUTPUT / f"conf_{threshold:.2f}.jpg"
            cv2.imwrite(str(output_path), annotated)

    print("\nDiagnostic images saved to:", OUTPUT)


if __name__ == "__main__":
    main()
