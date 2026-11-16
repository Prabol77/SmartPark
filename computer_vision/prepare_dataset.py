
from pathlib import Path
from collections import Counter
import csv
import random


# ==========================================
# CONFIGURATION
# ==========================================

DATASET_ROOT = Path(
    "datasets/public/PKLot/PKLot/PKLotSegmented"
)

OUTPUT_DIR = Path("datasets/processed")

LABEL_MAP = {
    "Empty": 0,
    "Occupied": 1
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

RANDOM_STATE = 42

SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15
}


# ==========================================
# COLLECT DATASET
# ==========================================

def collect_images():

    records = []

    print("Scanning PKLot dataset...")

    for image_path in DATASET_ROOT.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        parts = image_path.relative_to(
            DATASET_ROOT
        ).parts

        # Location / Weather / Date / Label / Filename

        if len(parts) != 5:
            continue

        location, weather, date, label, filename = parts

        if label not in LABEL_MAP:
            continue

        records.append({
            "path": str(image_path.resolve()),
            "label": LABEL_MAP[label],
            "class_name": label,
            "location": location,
            "weather": weather,
            "date": date,
            "group": f"{location}_{date}"
        })

    return records


# ==========================================
# GROUP-AWARE DATA SPLIT
# ==========================================


def create_splits(records):

    from collections import Counter, defaultdict
    import random

    # Group images by location and date.
    groups = defaultdict(list)

    for record in records:
        groups[record["group"]].append(record)

    group_items = list(groups.items())

    total_images = len(records)

    total_classes = Counter(
        record["label"] for record in records
    )

    ratios = {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15
    }

    targets = {}

    for name, ratio in ratios.items():

        targets[name] = {
            "total": total_images * ratio,
            0: total_classes[0] * ratio,
            1: total_classes[1] * ratio
        }

    def calculate_score(counts):

        score = 0.0

        for split in ratios:

            for key in ["total", 0, 1]:

                target = targets[split][key]

                if target == 0:
                    continue

                difference = (
                    counts[split][key] - target
                ) / target

                score += difference ** 2

        return score

    best_splits = None
    best_score = float("inf")

    # Multiple randomized greedy attempts.
    # Select the split with the best final distribution.

    for attempt in range(300):

        rng = random.Random(42 + attempt)

        shuffled = group_items.copy()
        rng.shuffle(shuffled)

        # Place larger groups first.
        shuffled.sort(
            key=lambda item: len(item[1]),
            reverse=True
        )

        current = {
            name: Counter()
            for name in ratios
        }

        candidate = {
            name: []
            for name in ratios
        }

        for group_id, images in shuffled:

            group_counts = Counter(
                record["label"] for record in images
            )

            group_size = len(images)

            best_split = None
            minimum_score = float("inf")

            for split in ratios:

                current[split]["total"] += group_size

                for label in [0, 1]:
                    current[split][label] += (
                        group_counts[label]
                    )

                score = calculate_score(current)

                # Undo temporary assignment.
                current[split]["total"] -= group_size

                for label in [0, 1]:
                    current[split][label] -= (
                        group_counts[label]
                    )

                if score < minimum_score:
                    minimum_score = score
                    best_split = split

            # Commit the group to the best split.
            candidate[best_split].extend(images)

            current[best_split]["total"] += group_size

            for label in [0, 1]:
                current[best_split][label] += (
                    group_counts[label]
                )

        final_score = calculate_score(current)

        if final_score < best_score:
            best_score = final_score
            best_splits = candidate

    return best_splits


    # Allocate complete groups to the split
    # with the largest remaining target deficit.

    for group_id, images in group_items:

        destination = max(
            targets,
            key=lambda name: targets[name] - counts[name]
        )

        splits[destination].extend(images)

        counts[destination] += len(images)

    return splits


# ==========================================
# SAVE CSV FILES
# ==========================================

def save_csv(filename, records):

    output_path = OUTPUT_DIR / filename

    fieldnames = [
        "path",
        "label",
        "class_name",
        "location",
        "weather",
        "date",
        "group"
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(records)


# ==========================================
# REPORT
# ==========================================

def report_split(name, records):

    distribution = Counter(
        record["class_name"]
        for record in records
    )

    print(f"\n========== {name.upper()} ==========")

    print("Total images:", f"{len(records):,}")

    print(
        "Empty:",
        f"{distribution.get('Empty', 0):,}"
    )

    print(
        "Occupied:",
        f"{distribution.get('Occupied', 0):,}"
    )

    print(
        "Unique groups:",
        len(set(record["group"] for record in records))
    )


# ==========================================
# MAIN
# ==========================================

def main():

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_ROOT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    records = collect_images()

    if not records:
        raise RuntimeError("No valid images found.")

    print("\n========== DATASET SUMMARY ==========")

    print("Total images:", f"{len(records):,}")

    distribution = Counter(
        record["class_name"]
        for record in records
    )

    print("Empty:", f"{distribution['Empty']:,}")
    print("Occupied:", f"{distribution['Occupied']:,}")

    splits = create_splits(records)

    # Verify no group leakage.

    group_sets = {
        name: set(record["group"] for record in data)
        for name, data in splits.items()
    }

    assert group_sets["train"].isdisjoint(
        group_sets["validation"]
    )

    assert group_sets["train"].isdisjoint(
        group_sets["test"]
    )

    assert group_sets["validation"].isdisjoint(
        group_sets["test"]
    )

    # Save datasets.

    save_csv("train.csv", splits["train"])
    save_csv("validation.csv", splits["validation"])
    save_csv("test.csv", splits["test"])

    # Print reports.

    for name, data in splits.items():
        report_split(name, data)

    print("\n========== VERIFICATION ==========")

    print("Group leakage: NONE")
    print("Dataset splitting: SUCCESS")

    print("\nFiles saved in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
