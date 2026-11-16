
import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "datasets" / "processed"
OUTPUT = PROCESSED / "location_holdout"

SEED = 42
TRAIN_RATIO = 0.80

INPUT_FILES = [
    PROCESSED / "train.csv",
    PROCESSED / "validation.csv",
    PROCESSED / "test.csv"
]


def main():

    print("Preparing UFPR04 unseen-location experiment...")

    records = []

    # Combine existing splits. They contain disjoint records.
    for file_path in INPUT_FILES:

        with open(file_path, encoding="utf-8", newline="") as f:
            records.extend(csv.DictReader(f))

    print("Total records:", len(records))

    # Group records by location and capture date.
    location_groups = defaultdict(lambda: defaultdict(list))

    for record in records:

        location = record["location"]
        group = record["group"]

        location_groups[location][group].append(record)

    # All UFPR04 groups are reserved for final testing.
    test_records = []

    for group_records in location_groups["UFPR04"].values():
        test_records.extend(group_records)

    # Other locations are eligible for training and validation.
    eligible_groups = []

    for location in ["PUC", "UFPR05"]:

        for group, group_records in location_groups[location].items():

            eligible_groups.append(
                (group, group_records)
            )

    rng = random.Random(SEED)
    rng.shuffle(eligible_groups)

    # Split entire groups, never individual images.
    target_train = int(
        sum(len(rows) for _, rows in eligible_groups)
        * TRAIN_RATIO
    )

    train_records = []
    val_records = []

    for group, group_records in eligible_groups:

        if len(train_records) < target_train:
            train_records.extend(group_records)
        else:
            val_records.extend(group_records)

    # Verify location separation.
    assert all(r["location"] == "UFPR04" for r in test_records)

    assert all(
        r["location"] in ["PUC", "UFPR05"]
        for r in train_records + val_records
    )

    train_groups = {r["group"] for r in train_records}
    val_groups = {r["group"] for r in val_records}
    test_groups = {r["group"] for r in test_records}

    assert not (train_groups & val_groups)
    assert not (train_groups & test_groups)
    assert not (val_groups & test_groups)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    splits = {
        "train.csv": train_records,
        "validation.csv": val_records,
        "test_ufpr04.csv": test_records
    }

    for filename, rows in splits.items():

        output_file = OUTPUT / filename

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "path",
                    "label",
                    "class_name",
                    "location",
                    "weather",
                    "date",
                    "group"
                ]
            )

            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSaved {filename}: {len(rows):,} images")

        print(
            "Locations:",
            sorted(set(r["location"] for r in rows))
        )

        print(
            "Groups:",
            len(set(r["group"] for r in rows))
        )

    print("\nGroup leakage: NONE")
    print("Location leakage: NONE")
    print("Location-held-out dataset preparation: SUCCESS")


if __name__ == "__main__":
    main()
