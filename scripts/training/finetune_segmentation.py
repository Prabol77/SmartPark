
from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    ROOT = Path(__file__).resolve().parents[2]

    MODEL_PATH = (
        ROOT / "models" / "parking_space_segmentation_best.pt"
    )

    DATASET_YAML = (
        ROOT / "datasets" / "parking_adaptation" / "data.yaml"
    )

    OUTPUT = ROOT / "runs" / "parking_segmentation_adaptation"

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not DATASET_YAML.exists():
        raise FileNotFoundError(DATASET_YAML)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Starting UFPR04 domain adaptation...")
    print("Initial checkpoint:", MODEL_PATH)

    model = YOLO(str(MODEL_PATH))

    model.train(
        data=str(DATASET_YAML),

        # Fine-tuning configuration
        epochs=15,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,

        # Conservative fine-tuning
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        patience=6,

        # Training settings
        amp=True,
        cache=False,
        close_mosaic=5,
        save=True,
        save_period=5,
        plots=True,

        # Separate experiment directory
        project=str(OUTPUT),
        name="ufpr04_finetune_v1",
        exist_ok=True,
        verbose=True
    )

    print("\nFine-tuning completed.")
    print(
        "Best checkpoint:",
        OUTPUT / "ufpr04_finetune_v1" / "weights" / "best.pt"
    )


if __name__ == "__main__":
    main()



