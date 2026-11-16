
from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets" / "parking_adaptation" / "images"

splits = ["train", "val", "test"]
hash_locations = defaultdict(list)

for split in splits:
    folder = DATASET / split

    if not folder.exists():
        raise FileNotFoundError(folder)

    images = [
        p for p in folder.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    print(f"{split}: {len(images)} images")

    for image in images:
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        hash_locations[digest].append((split, str(image)))

print("\nChecking exact duplicate images across splits...")

cross_split_duplicates = []

for digest, locations in hash_locations.items():
    unique_splits = {split for split, _ in locations}

    if len(unique_splits) > 1:
        cross_split_duplicates.append(locations)

print("Cross-split duplicate groups:", len(cross_split_duplicates))

for group in cross_split_duplicates[:20]:
    print("\nDuplicate group:")
    for split, path in group:
        print(split, path)

if cross_split_duplicates:
    print("\nWARNING: Investigate these overlaps before accepting test metrics.")
else:
    print("\nPASS: No exact image duplicates across train, val and test.")
