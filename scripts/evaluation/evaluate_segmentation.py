
from pathlib import Path
from ultralytics import YOLO
import torch


def main():
    ROOT = Path(__file__).resolve().parents[2]

    MODEL_PATH = (
        ROOT / "models" / "parking_space_segmentation_best.pt"
    )

    DATA_YAML = (
        ROOT / "datasets" / "parking_segmentation" / "data.yaml"
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not DATA_YAML.exists():
        raise FileNotFoundError(DATA_YAML)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")

    print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        plots=True,
        project=str(ROOT / "runs" / "parking_segmentation"),
        name="ufpr04_independent_test",
        exist_ok=True
    )

    print("\n========== UFPR04 TEST RESULTS ==========")
    print("Box mAP50:", metrics.box.map50)
    print("Box mAP50-95:", metrics.box.map)
    print("Mask mAP50:", metrics.seg.map50)
    print("Mask mAP50-95:", metrics.seg.map)
    print("Mask precision:", metrics.seg.mp)
    print("Mask recall:", metrics.seg.mr)


if __name__ == "__main__":
    main()
