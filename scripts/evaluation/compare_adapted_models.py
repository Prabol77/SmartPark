
from pathlib import Path
from ultralytics import YOLO
import torch

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "datasets" / "parking_adaptation" / "data.yaml"

MODELS = {
    "Original": ROOT / "models" / "parking_space_segmentation_best.pt",
    "Fine_Tuned": ROOT / "runs" / "parking_segmentation_adaptation" /
                  "ufpr04_finetune_v1" / "weights" / "best.pt",
}

OUTPUT = ROOT / "runs" / "final_model_comparison"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")

    if not DATA.exists():
        raise FileNotFoundError(DATA)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Evaluating both models on the same held-out UFPR04 test set.")
    print("Test split: 462 images")
    print("=" * 65)

    for name, checkpoint in MODELS.items():
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        print(f"\nEvaluating: {name}")
        print("Checkpoint:", checkpoint)

        model = YOLO(str(checkpoint))

        metrics = model.val(
            data=str(DATA),
            split="test",
            imgsz=640,
            batch=4,
            device=0,
            workers=0,
            plots=True,
            project=str(OUTPUT),
            name=name.lower(),
            exist_ok=True,
            verbose=True
        )

        print(f"\n{name} RESULTS")
        print("-" * 45)

        print("BOX METRICS")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall:    {metrics.box.mr:.4f}")
        print(f"mAP50:     {metrics.box.map50:.4f}")
        print(f"mAP50-95:  {metrics.box.map:.4f}")

        print("\nMASK METRICS")
        print(f"Precision: {metrics.seg.mp:.4f}")
        print(f"Recall:    {metrics.seg.mr:.4f}")
        print(f"mAP50:     {metrics.seg.map50:.4f}")
        print(f"mAP50-95:  {metrics.seg.map:.4f}")

        print("\nResults saved under:", OUTPUT / name.lower())

    print("\nCOMPARISON COMPLETED.")


if __name__ == "__main__":
    main()
