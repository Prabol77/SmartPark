
import csv
import random
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from occupancy_classifier import ParkingOccupancyCNN


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_CSV = ROOT_DIR / "datasets" / "processed" / "train.csv"
VAL_CSV = ROOT_DIR / "datasets" / "processed" / "validation.csv"
TEST_CSV = ROOT_DIR / "datasets" / "processed" / "test.csv"

MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_occupancy_model.pth"

# Smoke test settings
# Keep True for the first run.
# Change to False only after the smoke test succeeds.


SMOKE_TEST = True

MAX_TRAIN_SAMPLES = 50000
MAX_VAL_SAMPLES = 10000

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001
NUM_WORKERS = 0

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"

CLASS_NAMES = {
    0: "Empty",
    1: "Occupied"
}


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. STRATIFIED SAMPLE SELECTION
# ============================================================

def stratified_sample(records, max_samples, seed=42):

    if max_samples is None or len(records) <= max_samples:
        return records

    grouped = defaultdict(list)

    for record in records:
        label = int(record["label"])
        grouped[label].append(record)

    rng = random.Random(seed)

    total = len(records)
    selected = []

    quotas = {}

    # Calculate proportional quota for each class.
    for label, items in grouped.items():

        quota = round(
            max_samples * len(items) / total
        )

        quotas[label] = min(quota, len(items))

    # Adjust quotas to reach the requested sample count.
    while sum(quotas.values()) > max_samples:

        label = max(
            quotas,
            key=lambda x: quotas[x]
        )

        if quotas[label] > 0:
            quotas[label] -= 1

    while sum(quotas.values()) < max_samples:

        available = [
            label for label, items in grouped.items()
            if quotas[label] < len(items)
        ]

        if not available:
            break

        label = max(
            available,
            key=lambda x: len(grouped[x]) - quotas[x]
        )

        quotas[label] += 1

    # Select samples from each class.
    for label, items in grouped.items():

        rng.shuffle(items)

        selected.extend(
            items[:quotas[label]]
        )

    rng.shuffle(selected)

    return selected


# ============================================================
# 4. DATASET CLASS
# ============================================================

class ParkingDataset(Dataset):

    def __init__(
        self,
        csv_file,
        transform=None,
        max_samples=None
    ):

        self.csv_file = Path(csv_file)
        self.transform = transform

        if not self.csv_file.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found: {self.csv_file}"
            )

        with open(
            self.csv_file,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            self.records = list(reader)

        if not self.records:
            raise ValueError(
                f"No records found in {self.csv_file}"
            )

        # Verify required CSV columns.
        required_columns = {"path", "label"}

        if not required_columns.issubset(
            self.records[0].keys()
        ):

            raise ValueError(
                "CSV must contain 'path' and 'label' columns."
            )

        # Select a reproducible, class-balanced subset.
        self.records = stratified_sample(
            self.records,
            max_samples,
            SEED
        )

        # Validate labels.
        for record in self.records:

            label = int(record["label"])

            if label not in (0, 1):
                raise ValueError(
                    f"Invalid label {label} in {self.csv_file}"
                )

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image_path = Path(record["path"])
        label = int(record["label"])

        try:

            with Image.open(image_path) as image:

                image = image.convert("RGB")

        except Exception as error:

            raise RuntimeError(
                f"Could not load image: {image_path}"
            ) from error

        if self.transform:

            image = self.transform(image)

        return image, label


# ============================================================
# 5. IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

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

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ============================================================
# 6. METRICS
# ============================================================

def calculate_metrics(confusion_matrix):

    tn = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]

    fn = confusion_matrix[1][0]
    tp = confusion_matrix[1][1]

    total = tn + fp + fn + tp

    accuracy = (
        (tn + tp) / total
        if total else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def update_confusion_matrix(matrix, predictions, labels):

    for prediction, label in zip(
        predictions.tolist(),
        labels.tolist()
    ):

        matrix[label][prediction] += 1


# ============================================================
# 7. TRAINING FUNCTION
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

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    start_time = time.time()

    for batch_index, (images, labels) in enumerate(loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision on CUDA.
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_AMP
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        predictions = outputs.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_correct += (
            predictions == labels
        ).sum().item()

        total_images += batch_size

        if (batch_index + 1) % 25 == 0:

            print(
                f"Epoch [{epoch}/{EPOCHS}] "
                f"Batch [{batch_index + 1}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / total_images

    accuracy = total_correct / total_images

    elapsed = time.time() - start_time

    return avg_loss, accuracy, elapsed


# ============================================================
# 8. VALIDATION FUNCTION
# ============================================================

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total_images = 0

    confusion_matrix = [
        [0, 0],
        [0, 0]
    ]

    with torch.no_grad():

        for images, labels in loader:

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

                loss = criterion(
                    outputs,
                    labels
                )

            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            total_images += batch_size

            update_confusion_matrix(
                confusion_matrix,
                predictions,
                labels
            )

    avg_loss = total_loss / total_images

    metrics = calculate_metrics(
        confusion_matrix
    )

    return avg_loss, metrics, confusion_matrix


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

def print_results(title, loss, metrics, matrix):

    print("\n" + "=" * 55)

    print(title)

    print("=" * 55)

    print(f"Loss      : {loss:.4f}")

    print(
        f"Accuracy  : {metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision : {metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall    : {metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score  : {metrics['f1'] * 100:.2f}%"
    )

    print("\nConfusion Matrix:")

    print("                 Predicted")

    print("              Empty  Occupied")

    print(
        f"Actual Empty   {matrix[0][0]:6d}  {matrix[0][1]:8d}"
    )

    print(
        f"Actual Occupied{matrix[1][0]:6d}  {matrix[1][1]:8d}"
    )

    print("=" * 55)


# ============================================================
# 10. MAIN TRAINING PIPELINE
# ============================================================

def main():

    set_seed(SEED)

    print("\n" + "=" * 60)

    print("INTELLIGENT PARKING MANAGEMENT SYSTEM")

    print("Parking Occupancy CNN Training")

    print("=" * 60)

    print(f"Device       : {DEVICE}")

    print(f"Smoke test   : {SMOKE_TEST}")

    print(f"Image size   : {IMAGE_SIZE} x {IMAGE_SIZE}")

    print(f"Batch size   : {BATCH_SIZE}")

    print(f"Epochs       : {EPOCHS}")

    print(f"Learning rate: {LEARNING_RATE}")

    print(f"Mixed AMP    : {USE_AMP}")

    if torch.cuda.is_available():

        print(
            f"GPU          : {torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory   : "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = ParkingDataset(
        TRAIN_CSV,
        transform=train_transform,
        max_samples=(
            MAX_TRAIN_SAMPLES if SMOKE_TEST else None
        )
    )

    val_dataset = ParkingDataset(
        VAL_CSV,
        transform=val_transform,
        max_samples=(
            MAX_VAL_SAMPLES if SMOKE_TEST else None
        )
    )

    print(
        f"Training images  : {len(train_dataset):,}"
    )

    print(
        f"Validation images: {len(val_dataset):,}"
    )

    if not SMOKE_TEST:

        test_dataset = ParkingDataset(
            TEST_CSV,
            transform=val_transform
        )

        print(
            f"Test images      : {len(test_dataset):,}"
        )

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
    # Initialize model
    # --------------------------------------------------------

    print("\nInitializing CNN...")

    model = ParkingOccupancyCNN().to(DEVICE)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Model parameters: {total_parameters:,}"
    )

    # --------------------------------------------------------
    # Loss, optimizer and scaler
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_val_accuracy = -1.0

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_accuracy, elapsed = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            epoch
        )

        val_loss, val_metrics, val_matrix = evaluate(
            model,
            val_loader,
            criterion
        )

        print("\nEpoch Results")

        print("-" * 50)

        print(f"Epoch          : {epoch}/{EPOCHS}")

        print(f"Training Loss  : {train_loss:.4f}")

        print(
            f"Training Acc   : {train_accuracy * 100:.2f}%"
        )

        print(f"Validation Loss: {val_loss:.4f}")

        print(
            f"Validation Acc : "
            f"{val_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Epoch Time     : {elapsed:.2f} seconds"
        )

        # Save the best validation model.
        if val_metrics["accuracy"] > best_val_accuracy:

            best_val_accuracy = val_metrics["accuracy"]

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_accuracy": best_val_accuracy,
                "class_names": CLASS_NAMES,
                "image_size": IMAGE_SIZE
            }

            torch.save(
                checkpoint,
                MODEL_PATH
            )

            print(
                f"\nBest model saved: {MODEL_PATH}"
            )

        print_results(
            "VALIDATION RESULTS",
            val_loss,
            val_metrics,
            val_matrix
        )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    if not SMOKE_TEST:

        print("\nLoading best model for test evaluation...")

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=USE_AMP
        )

        test_loss, test_metrics, test_matrix = evaluate(
            model,
            test_loader,
            criterion
        )

        print_results(
            "FINAL TEST RESULTS",
            test_loss,
            test_metrics,
            test_matrix
        )

    print("\n" + "=" * 60)

    print("TRAINING COMPLETED")

    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")

    print(f"Model location: {MODEL_PATH}")

    if SMOKE_TEST:

        print(
            "NOTE: This was a smoke test, not final model evaluation."
        )

    print("=" * 60)


# ============================================================
# 11. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
