
from pathlib import Path
import cv2
import csv

from computer_vision.parking_pipeline import ParkingPipeline

ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = (
    ROOT / "datasets" / "parking_adaptation" / "images" / "test"
    / "UFPR04_007af75bbf_2012-12-12_11_25_06.jpg"
)

SEGMENTATION_MODEL = (
    ROOT / "runs" / "parking_segmentation_adaptation"
    / "ufpr04_finetune_v1" / "weights" / "best.pt"
)

OCCUPANCY_MODEL = (
    ROOT / "models" / "best_occupancy_ufpr04_holdout.pth"
)

OUTPUT_DIR = ROOT / "runs" / "ufpr04_pipeline_test"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(IMAGE_PATH)

    pipeline = ParkingPipeline(
        segmentation_checkpoint=SEGMENTATION_MODEL,
        occupancy_checkpoint=OCCUPANCY_MODEL,
        yolo_confidence=0.15,
        yolo_iou=0.50,
    )

    result = pipeline.analyse(image)
    rows = result["rows"]

    total = len(rows)
    empty = sum(r["prediction"] == "EMPTY" for r in rows)
    occupied = sum(r["prediction"] == "OCCUPIED" for r in rows)

    print("\nUFPR04 INTEGRATED PIPELINE TEST")
    print("-" * 45)
    print("Image:", IMAGE_PATH.name)
    print("YOLO detections:", result["detection_count"])
    print("Classified slots:", total)
    print("Empty:", empty)
    print("Occupied:", occupied)
    print(
        "Occupancy rate:",
        f"{occupied / total * 100:.2f}%" if total else "N/A"
    )
    print("Device:", result["device"])

    for row in rows:
        print(
            row["slot_id"],
            row["prediction"],
            row["confidence_percent"],
            row["confidence_status"],
        )

    output_image = OUTPUT_DIR / "ufpr04_annotated.jpg"

    cv2.imwrite(
        str(output_image),
        cv2.cvtColor(
            result["annotated_rgb"],
            cv2.COLOR_RGB2BGR,
        ),
    )

    output_csv = OUTPUT_DIR / "ufpr04_predictions.csv"

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nAnnotated image:", output_image)
    print("CSV:", output_csv)


if __name__ == "__main__":
    main()

