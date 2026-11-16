
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

MODEL = (
    ROOT / "runs" / "parking_segmentation_adaptation"
    / "ufpr04_finetune_v1" / "weights" / "best.pt"
)

TEST_IMAGES = ROOT / "datasets" / "parking_adaptation" / "images" / "test"

OUTPUT = ROOT / "runs" / "final_model_comparison" / "clean_predictions"


def main():
    model = YOLO(str(MODEL))

    images = sorted(
        p for p in TEST_IMAGES.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    selected = [
        images[round(i * (len(images) - 1) / 5)]
        for i in range(6)
    ]

    for conf in [0.25, 0.50, 0.70]:
        print(f"\nTesting confidence threshold: {conf}")

        model.predict(
            source=[str(p) for p in selected],
            imgsz=640,
            conf=conf,
            iou=0.5,
            device=0,
            retina_masks=True,
            show_labels=False,
            show_conf=False,
            save=True,
            project=str(OUTPUT),
            name=f"conf_{conf}",
            exist_ok=True,
            verbose=True
        )


if __name__ == "__main__":
    main()
