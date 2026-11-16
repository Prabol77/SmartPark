import csv
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from occupancy_classifier import ParkingOccupancyCNN


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TEST_CSV = (
    ROOT
    / "datasets"
    / "processed"
    / "location_holdout"
    / "test_ufpr04.csv"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "best_occupancy_ufpr04_holdout.pth"
)

IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"


# ============================================================
# DATASET
# ============================================================

class ParkingTestDataset(Dataset):

    def __init__(self, csv_path, transform):
        with open(csv_path, "r", encoding="utf-8") as file:
            self.records = list(csv.DictReader(file))

        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        with Image.open(record["path"]) as img:
            image = img.convert("RGB")

        image = self.transform(image)

        label = int(record["label"])

        location = record["location"]
        weather = record["weather"]

        return image, label, location, weather


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(matrix):
    tn = matrix[0][0]
    fp = matrix[0][1]
    fn = matrix[1][0]
    tp = matrix[1][1]

    total = tn + fp + fn + tp

    accuracy = (
        (tn + tp) / total if total else 0
    )

    precision = (
        tp / (tp + fp) if tp + fp else 0
    )

    recall = (
        tp / (tp + fn) if tp + fn else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return {
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def new_matrix():
    return [
        [0, 0],
        [0, 0]
    ]


def update_matrix(matrix, predictions, labels):
    for prediction, actual in zip(
        predictions,
        labels
    ):
        matrix[actual][prediction] += 1


# ============================================================
# REPORTING
# ============================================================

def print_report(title, results):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    print(
        f"{'Category':<18}"
        f"{'Images':>10}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    print("-" * 90)

    for category, matrix in sorted(results.items()):
        metrics = calculate_metrics(matrix)

        print(
            f"{category:<18}"
            f"{metrics['total']:>10,}"
            f"{metrics['accuracy'] * 100:>11.2f}%"
            f"{metrics['precision'] * 100:>11.2f}%"
            f"{metrics['recall'] * 100:>11.2f}%"
            f"{metrics['f1'] * 100:>11.2f}%"
        )

    print("=" * 90)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nPARKING OCCUPANCY - CONDITION EVALUATION")

    print("Device:", DEVICE)

    if not TEST_CSV.exists():
        raise FileNotFoundError(TEST_CSV)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
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

    # Verify that the final test contains ONLY UFPR04.
    test_locations = {
        record["location"]
        for record in dataset.records
    }

    print("Test locations:", sorted(test_locations))

    if test_locations != {"UFPR04"}:
        raise ValueError(
            f"Expected only UFPR04 test data, found: {test_locations}"
        )

    print("Unseen-location test verification: PASSED")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP
    )

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
    print("Test images:", len(dataset))

    # Results grouped independently by location and weather.
    location_results = defaultdict(new_matrix)
    weather_results = defaultdict(new_matrix)

    overall_matrix = new_matrix()

    print("\nEvaluating images...\n")

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images, labels, locations, weathers = batch

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

            predictions = predictions.cpu().tolist()
            actual_labels = labels.cpu().tolist()

            for prediction, actual, location, weather in zip(
                predictions,
                actual_labels,
                locations,
                weathers
            ):
                overall_matrix[actual][prediction] += 1
                location_results[location][actual][prediction] += 1
                weather_results[weather][actual][prediction] += 1

            if (batch_index + 1) % 100 == 0:
                processed = min(
                    (batch_index + 1) * BATCH_SIZE,
                    len(dataset)
                )

                print(
                    f"Processed {processed:,}/{len(dataset):,}"
                )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    overall = calculate_metrics(overall_matrix)

    print("\n" + "=" * 60)
    print("OVERALL TEST PERFORMANCE")
    print("=" * 60)

    print(f"Images    : {overall['total']:,}")
    print(f"Accuracy  : {overall['accuracy'] * 100:.2f}%")
    print(f"Precision : {overall['precision'] * 100:.2f}%")
    print(f"Recall    : {overall['recall'] * 100:.2f}%")
    print(f"F1 Score  : {overall['f1'] * 100:.2f}%")

    print("\nOverall Confusion Matrix:")
    print(overall_matrix)

    # --------------------------------------------------------
    # Location evaluation
    # --------------------------------------------------------

    print_report(
        "PER-LOCATION PERFORMANCE",
        location_results
    )

    # --------------------------------------------------------
    # Weather evaluation
    # --------------------------------------------------------

    print_report(
        "PER-WEATHER PERFORMANCE",
        weather_results
    )

    print("\nCondition evaluation completed.")


if __name__ == "__main__":
    main()