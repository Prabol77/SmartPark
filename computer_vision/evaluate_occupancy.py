
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from occupancy_classifier import ParkingOccupancyCNN


ROOT = Path(__file__).resolve().parents[1]

TEST_CSV = ROOT / "datasets" / "processed" / "test.csv"
MODEL_PATH = ROOT / "models" / "best_occupancy_model.pth"

IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"


class ParkingTestDataset(Dataset):

    def __init__(self, csv_path, transform):

        with open(csv_path, "r", encoding="utf-8") as f:
            self.records = list(csv.DictReader(f))

        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        with Image.open(record["path"]) as img:
            image = img.convert("RGB")

        image = self.transform(image)

        label = int(record["label"])

        return image, label


def main():

    print("=" * 55)
    print("PARKING OCCUPANCY - FINAL TEST EVALUATION")
    print("=" * 55)

    print("Device:", DEVICE)

    if not TEST_CSV.exists():
        raise FileNotFoundError(TEST_CSV)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    transform = transforms.Compose([

        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])

    dataset = ParkingTestDataset(
        TEST_CSV,
        transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP
    )

    print("Test images:", len(dataset))

    print("\nLoading best checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model = ParkingOccupancyCNN().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])

    print(
        "Best validation accuracy:",
        f"{checkpoint['best_val_accuracy'] * 100:.2f}%"
    )

    print("\nEvaluating test dataset...")

    confusion = [
        [0, 0],
        [0, 0]
    ]

    total = 0
    correct = 0

    with torch.no_grad():

        for batch_index, (images, labels) in enumerate(loader):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP
            ):

                outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for prediction, actual in zip(
                predictions.tolist(),
                labels.tolist()
            ):

                confusion[actual][prediction] += 1

            if (batch_index + 1) % 100 == 0:

                print(
                    f"Processed {total:,}/{len(dataset):,} images"
                )

    tn = confusion[0][0]
    fp = confusion[0][1]
    fn = confusion[1][0]
    tp = confusion[1][1]

    accuracy = correct / total

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print("\n" + "=" * 55)
    print("FINAL TEST RESULTS")
    print("=" * 55)

    print(f"Total images : {total:,}")
    print(f"Accuracy     : {accuracy * 100:.2f}%")
    print(f"Precision    : {precision * 100:.2f}%")
    print(f"Recall       : {recall * 100:.2f}%")
    print(f"F1 Score     : {f1 * 100:.2f}%")

    print("\nConfusion Matrix")
    print("                 Predicted")
    print("              Empty  Occupied")

    print(
        f"Actual Empty   {tn:6d}  {fp:8d}"
    )

    print(
        f"Actual Occupied{fn:6d}  {tp:8d}"
    )

    print("=" * 55)


if __name__ == "__main__":
    main()
