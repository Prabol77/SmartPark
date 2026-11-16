
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

MODEL = ROOT / "runs" / "parking_segmentation_adaptation" / \
    "ufpr04_finetune_v1" / "weights" / "best.pt"

TEST_IMAGES = ROOT / "datasets" / "parking_adaptation" / \
    "images" / "test"

OUTPUT = ROOT / "runs" / "final_model_comparison" / "visual_inspection"

def main():
    if not MODEL.exists():
        raise FileNotFoundError(MODEL)

    images = sorted(
        p for p in TEST_IMAGES.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if len(images) < 12:
        raise RuntimeError("Insufficient test images.")

    # Select 12 images distributed across the test set.
    selected = [
        images[round(i * (len(images) - 1) / 11)]
        for i in range(12)
    ]

    model = YOLO(str(MODEL))

    print("Selected held-out test images:")
    for path in selected:
        print(path.name)

    model.predict(
        source=[str(p) for p in selected],
        imgsz=640,
        conf=0.25,
        device=0,
        retina_masks=True,
        save=True,
        project=str(OUTPUT.parent),
        name=OUTPUT.name,
        exist_ok=True,
        verbose=True
    )

    print("\nSaved visual predictions to:", OUTPUT)


if __name__ == "__main__":
    main()
