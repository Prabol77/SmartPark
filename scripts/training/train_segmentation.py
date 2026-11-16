
from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parents[2]
    dataset_yaml = project_root / "datasets" / "parking_segmentation" / "data.yaml"

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check your PyTorch GPU installation.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print("Dataset:", dataset_yaml)

    model = YOLO("yolov8n-seg.pt")

    model.train(
        data=str(dataset_yaml),
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        patience=12,
        amp=True,
        cache=False,
        project=str(project_root / "runs" / "parking_segmentation"),
        name="yolov8n_pklot",
        exist_ok=True,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()



