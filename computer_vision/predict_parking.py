"""
Run CNN occupancy inference on manually extracted parking-slot crops.

Run from the project root:
    python computer_vision/predict_parking.py

Optional:
    python computer_vision/predict_parking.py --map datasets/processed/parking_slot_maps/parking_sample_slots.json
    python computer_vision/predict_parking.py --empty-index 0 --occupied-index 1

IMPORTANT:
- The default class mapping assumes training used alphabetical folder labels:
      Empty = 0, Occupied = 1
  Verify this against the class_to_idx mapping from your training dataset.
- This script reports model predictions; it does not assume the PKLot test
  accuracy transfers unchanged to a new camera/image.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# Keep this architecture identical to occupancy_classifier.py.
class ParkingOccupancyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict empty/occupied status for mapped parking slots."
    )
    parser.add_argument(
        "--map",
        default="datasets/processed/parking_slot_maps/parking_sample_slots.json",
        help="Path to the slot-map JSON created by parking_slots.py",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/best_occupancy_ufpr04_holdout.pth",
        help="Trained CNN checkpoint",
    )
    parser.add_argument(
        "--crops",
        default=None,
        help="Crop directory. Defaults to datasets/processed/parking_slot_crops/<image_stem>",
    )
    parser.add_argument(
        "--output",
        default="runs/parking_inference",
        help="Directory for CSV, JSON and annotated image outputs",
    )
    parser.add_argument(
        "--empty-index",
        type=int,
        default=0,
        help="Output index corresponding to Empty (verify against training class_to_idx)",
    )
    parser.add_argument(
        "--occupied-index",
        type=int,
        default=1,
        help="Output index corresponding to Occupied (verify against training class_to_idx)",
    )
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            # A plain state_dict is itself a dictionary of tensors.
            state_dict = checkpoint
    else:
        raise TypeError("Unsupported checkpoint format: expected a state dictionary.")

    # Handle checkpoints saved from nn.DataParallel.
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value
                      for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return checkpoint


def main():
    args = parse_args()

    map_path = Path(args.map)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output)

    if not map_path.is_file():
        raise FileNotFoundError(f"Slot map not found: {map_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    if args.empty_index == args.occupied_index:
        raise ValueError("Empty and Occupied class indices must be different.")

    with map_path.open("r", encoding="utf-8") as file:
        slot_map = json.load(file)

    source_image_path = Path(slot_map["source_image"])
    if not source_image_path.is_file():
        raise FileNotFoundError(
            f"Source parking image recorded in slot map was not found: {source_image_path}"
        )

    image_stem = source_image_path.stem
    crops_dir = (
        Path(args.crops)
        if args.crops
        else Path("datasets/processed/parking_slot_crops") / image_stem
    )
    if not crops_dir.is_dir():
        raise FileNotFoundError(f"Crop directory not found: {crops_dir}")

    if args.empty_index not in (0, 1) or args.occupied_index not in (0, 1):
        raise ValueError("For this two-class model, class indices must be 0 or 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = ParkingOccupancyCNN(num_classes=2)
    checkpoint = load_checkpoint(model, checkpoint_path, device)

    # If the checkpoint stores class_to_idx, display it to help verify mapping.
    if isinstance(checkpoint, dict):
        class_to_idx = checkpoint.get("class_to_idx")
        if class_to_idx:
            print(f"Checkpoint class_to_idx: {class_to_idx}")
            if "Empty" in class_to_idx and "Occupied" in class_to_idx:
                if (class_to_idx["Empty"] != args.empty_index or
                        class_to_idx["Occupied"] != args.occupied_index):
                    raise ValueError(
                        "The supplied class indices do not match checkpoint class_to_idx. "
                        "Rerun with --empty-index and --occupied-index matching that mapping."
                    )

    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    predictions = []
    slots = slot_map.get("slots", [])
    if not slots:
        raise ValueError("No slots were found in the slot-map JSON.")

    with torch.inference_mode():
        for slot in slots:
            slot_id = slot["slot_id"]
            crop_path = crops_dir / f"{slot_id}.jpg"

            if not crop_path.is_file():
                print(f"WARNING: missing crop for {slot_id}: {crop_path}")
                continue

            try:
                image = Image.open(crop_path).convert("RGB")
                tensor = preprocess(image).unsqueeze(0).to(device)
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

                predicted_index = int(np.argmax(probabilities))
                if predicted_index == args.empty_index:
                    predicted_label = "EMPTY"
                elif predicted_index == args.occupied_index:
                    predicted_label = "OCCUPIED"
                else:
                    raise ValueError(f"Unexpected predicted class index: {predicted_index}")

                confidence = float(probabilities[predicted_index])
                predictions.append({
                    "slot_id": slot_id,
                    "prediction": predicted_label,
                    "confidence": round(confidence, 6),
                    "empty_probability": round(float(probabilities[args.empty_index]), 6),
                    "occupied_probability": round(float(probabilities[args.occupied_index]), 6),
                    "crop_path": str(crop_path),
                })

                print(
                    f"{slot_id}: {predicted_label:<8} "
                    f"confidence={confidence * 100:6.2f}%"
                )

            except Exception as exc:
                print(f"ERROR processing {slot_id}: {exc}")

    if not predictions:
        raise RuntimeError("No crops were successfully processed.")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{image_stem}_predictions.csv"
    json_path = output_dir / f"{image_stem}_predictions.json"
    annotated_path = output_dir / f"{image_stem}_annotated.jpg"

    empty_count = sum(item["prediction"] == "EMPTY" for item in predictions)
    occupied_count = sum(item["prediction"] == "OCCUPIED" for item in predictions)

    # Save detailed CSV.
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)

    result = {
        "source_image": str(source_image_path),
        "slot_map": str(map_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "total_slots_predicted": len(predictions),
        "empty_slots": empty_count,
        "occupied_slots": occupied_count,
        "occupancy_rate_percent": round(occupied_count / len(predictions) * 100, 2),
        "class_indices": {
            "empty": args.empty_index,
            "occupied": args.occupied_index,
        },
        "predictions": predictions,
        "evaluation_note": (
            "These are predictions on this input image. The held-out PKLot test "
            "metrics do not guarantee the same accuracy on this camera/image."
        ),
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    # Draw the slot polygons and labels on the original parking image.
    annotated = cv2.imread(str(source_image_path))
    if annotated is None:
        raise RuntimeError(f"Could not read source image: {source_image_path}")

    prediction_by_id = {item["slot_id"]: item for item in predictions}
    for slot in slots:
        slot_id = slot["slot_id"]
        item = prediction_by_id.get(slot_id)
        if item is None:
            continue

        points = np.array(slot["points"], dtype=np.int32).reshape((-1, 1, 2))
        is_occupied = item["prediction"] == "OCCUPIED"
        color = (0, 0, 255) if is_occupied else (0, 200, 0)
        cv2.polylines(annotated, [points], True, color, 2)

        center = np.mean(points.reshape(-1, 2), axis=0).astype(int)
        label = f"{slot_id} {item['prediction']} {item['confidence'] * 100:.0f}%"
        cv2.putText(
            annotated, label, (int(center[0]), int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA
        )

    summary = f"Empty: {empty_count} | Occupied: {occupied_count} | Total: {len(predictions)}"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 36), (25, 25, 25), -1)
    cv2.putText(
        annotated, summary, (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.imwrite(str(annotated_path), annotated)

    print("\n========== PARKING OCCUPANCY SUMMARY ==========")
    print(f"Slots predicted : {len(predictions)} / {len(slots)}")
    print(f"Empty           : {empty_count}")
    print(f"Occupied        : {occupied_count}")
    print(f"Occupancy rate  : {occupied_count / len(predictions) * 100:.2f}%")
    print("\nSaved outputs:")
    print(f"CSV       : {csv_path.resolve()}")
    print(f"JSON      : {json_path.resolve()}")
    print(f"Annotated : {annotated_path.resolve()}")
    print("\nReminder: verify Empty/Occupied class index order against training.")


if __name__ == "__main__":
    main()
