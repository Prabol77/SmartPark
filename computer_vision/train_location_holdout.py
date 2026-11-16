import csv
import random
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from occupancy_classifier import ParkingOccupancyCNN


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "datasets" / "processed" / "location_holdout"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "validation.csv"
TEST_CSV = DATA_DIR / "test_ufpr04.csv"

MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "best_occupancy_ufpr04_holdout.pth"

# Start with a pipeline smoke test.
SMOKE_TEST = False

MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

IMAGE_SIZE = 128
BATCH_SIZE = 16 if SMOKE_TEST else 32
EPOCHS = 1 if SMOKE_TEST else 5

LEARNING_RATE = 0.001
NUM_WORKERS = 0
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ============================================================
# STRATIFIED SAMPLING
# ============================================================

def stratified_sample(records, max_samples):
    if max_samples is None or len(records) <= max_samples:
        return records

    grouped = {}

    for record in records:
        label = int(record["label"])
        grouped.setdefault(label, []).append(record)

    rng = random.Random(SEED)

    total = len(records)

    quotas = {
        label: round(max_samples * len(items) / total)
        for label, items in grouped.items()
    }

    while sum(quotas.values()) > max_samples:
        label = max(quotas, key=quotas.get)

        if quotas[label] > 0:
            quotas[label] -= 1

    while sum(quotas.values()) < max_samples:
        available = [
            label for label in grouped
            if quotas[label] < len(grouped[label])
        ]

        if not available:
            break

        label = max(
            available,
            key=lambda x: len(grouped[x]) - quotas[x]
        )

        quotas[label] += 1

    selected = []

    for label, items in grouped.items():
        rng.shuffle(items)
        selected.extend(items[:quotas[label]])

    rng.shuffle(selected)

    return selected


# ============================================================
# DATASET
# ============================================================

class ParkingDataset(Dataset):

    def __init__(self, csv_path, transform, max_samples=None):
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        with open(
            csv_path,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:
            self.records = list(csv.DictReader(file))

        if not self.records:
            raise ValueError(f"No records in {csv_path}")

        self.transform = transform

        self.records = stratified_sample(
            self.records,
            max_samples
        )

        self.csv_path = csv_path

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image_path = Path(record["path"])
        label = int(record["label"])

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")

        except Exception as error:
            raise RuntimeError(
                f"Could not load image: {image_path}"
            ) from error

        image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# EVALUATION METRICS
# ============================================================

def calculate_metrics(matrix):
    tn, fp = matrix[0]
    fn, tp = matrix[1]

    total = tn + fp + fn + tp

    accuracy = (tn + tp) / total if total else 0

    precision = tp / (tp + fp) if tp + fp else 0

    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    epoch
):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    start = time.time()

    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_AMP
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predictions = outputs.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size

        if (batch_index + 1) % 50 == 0:
            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Batch {batch_index + 1}/{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    return (
        total_loss / total,
        correct / total,
        time.time() - start
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0
    total = 0

    matrix = [
        [0, 0],
        [0, 0]
    ]

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            for prediction, actual in zip(
                predictions.tolist(),
                labels.tolist()
            ):
                matrix[actual][prediction] += 1

    metrics = calculate_metrics(matrix)

    return total_loss / total, metrics, matrix


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed()

    print("\n" + "=" * 60)
    print("UFPR04 UNSEEN-LOCATION EXPERIMENT")
    print("=" * 60)

    print("Device:", DEVICE)
    print("GPU:", torch.cuda.get_device_name(0)
          if torch.cuda.is_available() else "CPU")
    print("Smoke test:", SMOKE_TEST)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading training data...")

    train_dataset = ParkingDataset(
        TRAIN_CSV,
        train_transform,
        MAX_TRAIN_SAMPLES if SMOKE_TEST else None
    )

    val_dataset = ParkingDataset(
        VAL_CSV,
        val_transform,
        MAX_VAL_SAMPLES if SMOKE_TEST else None
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    # --------------------------------------------------------
    # Verify locations in the CSVs
    # --------------------------------------------------------

    for name, path in [
        ("Training", TRAIN_CSV),
        ("Validation", VAL_CSV)
    ]:
        with open(path, encoding="utf-8") as f:
            records = list(csv.DictReader(f))

        locations = set(r["location"] for r in records)

        if not locations.issubset({"PUC", "UFPR05"}):
            raise ValueError(
                f"Unexpected location in {name}: {locations}"
            )

        print(f"{name} locations:", sorted(locations))

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP
    )

    # --------------------------------------------------------
    # Fresh model initialization
    # --------------------------------------------------------

    print("\nInitializing CNN...")
    model = ParkingOccupancyCNN().to(DEVICE)

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Resume training from the latest checkpoint
    # --------------------------------------------------------

    LAST_MODEL_PATH = MODEL_DIR / "last_occupancy_ufpr04_holdout.pth"

    # Prefer the latest checkpoint; otherwise use the verified
    # best checkpoint from Epoch 2.
    RESUME_PATH = (
        LAST_MODEL_PATH
        if LAST_MODEL_PATH.exists()
        else MODEL_PATH
    )

    start_epoch = 1
    best_accuracy = -1.0

    if RESUME_PATH.exists():
        print("\nLoading checkpoint:", RESUME_PATH)

        checkpoint = torch.load(
            RESUME_PATH,
            map_location=DEVICE,
            weights_only=False
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_accuracy = checkpoint.get("best_val_accuracy", -1.0)

        # Restore AMP scaler if a previous checkpoint contains it.
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print("AMP scaler state restored.")
        else:
            print("AMP scaler state unavailable; starting a fresh scaler.")

        print("Resuming from epoch:", start_epoch)
        print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")

    else:
        print("No checkpoint found. Starting training from scratch.")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss, train_acc, duration = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            epoch
        )

        val_loss, val_metrics, matrix = evaluate(
            model,
            val_loader,
            criterion
        )

        print("\n" + "=" * 55)
        print(f"EPOCH {epoch}/{EPOCHS}")
        print("=" * 55)

        print(f"Training loss: {train_loss:.4f}")
        print(f"Training accuracy: {train_acc * 100:.2f}%")
        print(f"Validation loss: {val_loss:.4f}")
        print(
            f"Validation accuracy: "
            f"{val_metrics['accuracy'] * 100:.2f}%"
        )
        print(f"Validation F1: {val_metrics['f1'] * 100:.2f}%")
        print(f"Duration: {duration:.2f} seconds")

        # Save the best-performing validation model.
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_accuracy": best_accuracy,
                    "class_names": {
                        0: "Empty",
                        1: "Occupied"
                    },
                    "image_size": IMAGE_SIZE,
                    "experiment": "UFPR04 unseen-location"
                },
                MODEL_PATH
            )

            print("\nBest model saved:", MODEL_PATH)

        # Save the latest state after EVERY epoch, even if
        # validation accuracy did not improve.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_accuracy": best_accuracy,
                "class_names": {
                    0: "Empty",
                    1: "Occupied"
                },
                "image_size": IMAGE_SIZE,
                "experiment": "UFPR04 unseen-location"
            },
            LAST_MODEL_PATH
        )

        print("Latest checkpoint saved:", LAST_MODEL_PATH)

        print("\nValidation confusion matrix:")
        print(matrix)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("Best validation accuracy:", f"{best_accuracy * 100:.2f}%")
    print("Best checkpoint:", MODEL_PATH)
    print("Latest checkpoint:", LAST_MODEL_PATH)

    if SMOKE_TEST:
        print("This was a pipeline smoke test.")
        print("Do not use this checkpoint for final reporting.")

    print("=" * 60)


if __name__ == "__main__":
    main()