
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FOLDER = ROOT / "datasets" / "processed" / "location_holdout"

SEED = 42
TRAIN_RATIO = 0.80


def load_csv(filename):

    with open(FOLDER / filename, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(filename, records):

    output = FOLDER / filename

    with open(output, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path", "label", "class_name",
                "location", "weather", "date", "group"
            ]
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {filename}: {len(records):,} images")


def main():

    # Recombine only PUC and UFPR05.
    original_train = load_csv("train.csv")
    original_val = load_csv("validation.csv")

    eligible = original_train + original_val

    assert all(
        r["location"] in ("PUC", "UFPR05")
        for r in eligible
    )

    # Group entire capture dates.
    groups = {}

    for record in eligible:

        groups.setdefault(record["group"], []).append(record)

    # Shuffle groups reproducibly.
    group_items = list(groups.items())

    random.Random(SEED).shuffle(group_items)

    total_images = sum(
        len(records) for _, records in group_items
    )

    target_train = int(total_images * TRAIN_RATIO)

    train_records = []
    val_records = []

    for group, records in group_items:

        if len(train_records) < target_train:
            train_records.extend(records)
        else:
            val_records.extend(records)

    # Preserve the original UFPR04 test set.
    test_records = load_csv("test_ufpr04.csv")

    assert all(
        r["location"] == "UFPR04"
        for r in test_records
    )

    train_groups = {r["group"] for r in train_records}
    val_groups = {r["group"] for r in val_records}
    test_groups = {r["group"] for r in test_records}

    assert not train_groups & val_groups
    assert not train_groups & test_groups
    assert not val_groups & test_groups

    assert {
        r["location"] for r in train_records
    } == {"PUC", "UFPR05"}

    assert {
        r["location"] for r in val_records
    } == {"PUC", "UFPR05"}

    save_csv("train.csv", train_records)
    save_csv("validation.csv", val_records)

    print("\nFINAL SPLIT SUMMARY")

    for name, records in [
        ("TRAIN", train_records),
        ("VALIDATION", val_records),
        ("TEST", test_records)
    ]:

        print(f"\n{name}: {len(records):,} images")

        for location in sorted({
            r["location"] for r in records
        }):

            count = sum(
                r["location"] == location
                for r in records
            )

            groups_count = len({
                r["group"] for r in records
                if r["location"] == location
            })

            print(
                f"  {location}: {count:,} images, "
                f"{groups_count} groups"
            )

    print("\nGroup leakage: NONE")
    print("Location leakage: NONE")
    print("Split correction: SUCCESS")


if __name__ == "__main__":
    main()
