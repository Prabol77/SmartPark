
from pathlib import Path
from collections import Counter
import csv
import os
import random
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets" / "parking_segmentation"
DEST = ROOT / "datasets" / "parking_adaptation"

SEED = 42
rng = random.Random(SEED)

report_path = SOURCE / "conversion_summary.csv"

if not report_path.exists():
    raise FileNotFoundError(report_path)

if DEST.exists():
    raise RuntimeError(
        f"Destination already exists: {DEST}\n"
        "Inspect or rename it before running this script."
    )

with report_path.open(newline="", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

ufpr04_dates = sorted({
    re.search(r"\d{4}-\d{2}-\d{2}", row["image"]).group(0)
    for row in records
    if row["location"] == "UFPR04"
})

assert len(ufpr04_dates) == 28, (
    f"Expected 28 dates, found {len(ufpr04_dates)}"
)

rng.shuffle(ufpr04_dates)

adapt_train_dates = set(ufpr04_dates[:20])
adapt_val_dates = set(ufpr04_dates[20:24])
adapt_test_dates = set(ufpr04_dates[24:])

assert not (adapt_train_dates & adapt_val_dates)
assert not (adapt_train_dates & adapt_test_dates)
assert not (adapt_val_dates & adapt_test_dates)

def destination_split(row):
    if row["location"] != "UFPR04":
        # Retain original training data.
        # Original validation data stays out of adaptation training.
        return row["split"]

    match = re.search(r"\d{4}-\d{2}-\d{2}", row["image"])
    if not match:
        raise ValueError(f"No date in {row['image']}")

    date = match.group(0)

    if date in adapt_train_dates:
        return "train"
    if date in adapt_val_dates:
        return "val"
    return "test"


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


counts = Counter()

for row in records:
    image_name = row["image"]
    original_split = row["split"]
    new_split = destination_split(row)

    source_image = SOURCE / "images" / original_split / image_name
    source_label = SOURCE / "labels" / original_split / (
        Path(image_name).stem + ".txt"
    )

    dest_image = DEST / "images" / new_split / image_name
    dest_label = DEST / "labels" / new_split / (
        Path(image_name).stem + ".txt"
    )

    if not source_image.exists() or not source_label.exists():
        raise FileNotFoundError(
            f"Missing source pair: {image_name}"
        )

    link_or_copy(source_image, dest_image)
    link_or_copy(source_label, dest_label)

    counts[new_split] += 1

yaml_path = DEST / "data.yaml"
yaml_path.write_text(
    f"""path: {DEST.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: parking_space
""",
    encoding="utf-8"
)

manifest = DEST / "ufpr04_date_split.txt"
manifest.write_text(
    "TRAIN DATES:\n" +
    "\n".join(sorted(adapt_train_dates)) +
    "\n\nVALIDATION DATES:\n" +
    "\n".join(sorted(adapt_val_dates)) +
    "\n\nTEST DATES:\n" +
    "\n".join(sorted(adapt_test_dates)) +
    "\n",
    encoding="utf-8"
)

print("\n========== ADAPTATION DATASET ==========")
for split in ("train", "val", "test"):
    print(f"{split}: {counts[split]} images")

print("\nUFPR04 training dates:", sorted(adapt_train_dates))
print("UFPR04 validation dates:", sorted(adapt_val_dates))
print("UFPR04 test dates:", sorted(adapt_test_dates))
print("\nDataset YAML:", yaml_path)
print("Date manifest:", manifest)



