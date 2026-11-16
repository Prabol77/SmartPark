
from pathlib import Path
import cv2
import csv
import json

from computer_vision.parking_pipeline import ParkingPipeline

ROOT = Path(__file__).resolve().parents[2]

SEGMENTATION_MODEL = (
    ROOT / "runs" / "parking_segmentation_adaptation"
    / "ufpr04_finetune_v1" / "weights" / "best.pt"
)

OCCUPANCY_MODEL = (
    ROOT / "models" / "best_occupancy_ufpr04_holdout.pth"
)

IMAGE_PATH = ROOT / "datasets" / "public" / "parking_sample.jpg"

OUTPUT_DIR = ROOT / "runs" / "integrated_pipeline_test"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(IMAGE_PATH)

    print("=" * 55)
    print("INTELLIGENT PARKING - INTEGRATION TEST")
    print("=" * 55)

    pipeline = ParkingPipeline(
        segmentation_checkpoint=SEGMENTATION_MODEL,
        occupancy_checkpoint=OCCUPANCY_MODEL,
        yolo_confidence=0.15,
        yolo_iou=0.50,
    )

    print("\nRunning YOLO segmentation and CNN classification...")

    result = pipeline.analyse(image)

    rows = result["rows"]

    total = len(rows)
    empty = sum(r["prediction"] == "EMPTY" for r in rows)
    occupied = sum(r["prediction"] == "OCCUPIED" for r in rows)

    high = sum(r["confidence_status"] == "HIGH_CONFIDENCE" for r in rows)
    review = sum(r["confidence_status"] == "REVIEW" for r in rows)
    uncertain = sum(r["confidence_status"] == "UNCERTAIN" for r in rows)

    occupancy_rate = (
        occupied / total * 100 if total else 0
    )

    print("\nRESULTS")
    print("-" * 55)
    print("YOLO detections:", result["detection_count"])
    print("Successfully classified slots:", total)
    print("Empty:", empty)
    print("Occupied:", occupied)
    print(f"Occupancy rate: {occupancy_rate:.2f}%")
    print("High confidence:", high)
    print("Review:", review)
    print("Uncertain:", uncertain)
    print("CNN device:", result["device"])

    csv_path = OUTPUT_DIR / "parking_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = OUTPUT_DIR / "parking_results.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "total_slots": total,
                "empty_slots": empty,
                "occupied_slots": occupied,
                "occupancy_rate_percent": round(occupancy_rate, 2),
                "high_confidence": high,
                "review": review,
                "uncertain": uncertain,
                "device": result["device"],
                "slots": rows,
            },
            file,
            indent=4,
        )

    image_path = OUTPUT_DIR / "annotated_parking.jpg"

    cv2.imwrite(
        str(image_path),
        cv2.cvtColor(
            result["annotated_rgb"],
            cv2.COLOR_RGB2BGR,
        ),
    )

    print("\nSAVED FILES")
    print("CSV:", csv_path)
    print("JSON:", json_path)
    print("Annotated image:", image_path)

    print("\nPIPELINE TEST COMPLETED.")


if __name__ == "__main__":
    main()
