
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / "datasets" / "parking_segmentation"
destination = ROOT / "parking_segmentation.zip"

if not source.exists():
    raise FileNotFoundError(source)

files = [p for p in source.rglob("*") if p.is_file()]

print(f"Files to archive: {len(files)}")

with zipfile.ZipFile(
    destination,
    "w",
    compression=zipfile.ZIP_STORED
) as archive:
    for index, file in enumerate(files, 1):
        archive.write(file, file.relative_to(source.parent))

        if index % 500 == 0:
            print(f"Archived {index}/{len(files)}")

print(f"\nArchive created: {destination.resolve()}")
print(f"Size: {destination.stat().st_size / (1024**3):.2f} GB")
