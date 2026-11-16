
import csv
from pathlib import Path


BASELINE = Path(
    "runs/parking_inference/baseline/parking_sample_predictions.csv"
)

IMPROVED = Path(
    "runs/parking_inference/parking_sample_predictions.csv"
)

OUTPUT_DIR = Path("runs/parking_inference/comparison")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "crop_prediction_comparison.csv"


def load_predictions(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    return {row["slot_id"]: row for row in rows}


baseline = load_predictions(BASELINE)
improved = load_predictions(IMPROVED)

if set(baseline) != set(improved):
    missing_new = sorted(set(baseline) - set(improved))
    missing_old = sorted(set(improved) - set(baseline))

    raise ValueError(
        f"Slot mismatch. Missing in improved: {missing_new}; "
        f"Missing in baseline: {missing_old}"
    )


comparison = []

for slot_id in sorted(
    baseline,
    key=lambda value: int(value.replace("P", ""))
):
    old = baseline[slot_id]
    new = improved[slot_id]

    old_conf = float(old["confidence"])
    new_conf = float(new["confidence"])

    old_empty = float(old["empty_probability"])
    new_empty = float(new["empty_probability"])

    old_occupied = float(old["occupied_probability"])
    new_occupied = float(new["occupied_probability"])

    old_prediction = old["prediction"]
    new_prediction = new["prediction"]

    comparison.append({
        "slot_id": slot_id,
        "original_prediction": old_prediction,
        "improved_prediction": new_prediction,
        "prediction_changed": old_prediction != new_prediction,
        "original_confidence": round(old_conf, 6),
        "improved_confidence": round(new_conf, 6),
        "confidence_change": round(new_conf - old_conf, 6),
        "original_empty_probability": round(old_empty, 6),
        "improved_empty_probability": round(new_empty, 6),
        "empty_probability_change": round(new_empty - old_empty, 6),
        "original_occupied_probability": round(old_occupied, 6),
        "improved_occupied_probability": round(new_occupied, 6),
        "occupied_probability_change": round(
            new_occupied - old_occupied, 6
        ),
        "crop_path": new["crop_path"]
    })


with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=comparison[0].keys()
    )
    writer.writeheader()
    writer.writerows(comparison)


def summarize(rows, key):
    empty = sum(row[key] == "EMPTY" for row in rows)
    occupied = sum(row[key] == "OCCUPIED" for row in rows)

    return empty, occupied


old_empty, old_occupied = summarize(
    [baseline[k] for k in baseline], "prediction"
)

new_empty, new_occupied = summarize(
    [improved[k] for k in improved], "prediction"
)

changed = [
    row for row in comparison
    if row["prediction_changed"]
]

mean_old_conf = sum(
    float(row["confidence"]) for row in baseline.values()
) / len(baseline)

mean_new_conf = sum(
    float(row["confidence"]) for row in improved.values()
) / len(improved)


print("\n========== CROP PREDICTION COMPARISON ==========")

print(f"Total matched slots : {len(comparison)}")

print("\nOriginal crops:")
print(f"Empty               : {old_empty}")
print(f"Occupied            : {old_occupied}")
print(f"Occupancy rate      : {old_occupied / len(baseline) * 100:.2f}%")
print(f"Mean confidence     : {mean_old_conf * 100:.2f}%")

print("\nExpanded crops:")
print(f"Empty               : {new_empty}")
print(f"Occupied            : {new_occupied}")
print(f"Occupancy rate      : {new_occupied / len(improved) * 100:.2f}%")
print(f"Mean confidence     : {mean_new_conf * 100:.2f}%")

print("\nPrediction changes:")
print(f"Changed slots       : {len(changed)}")
print(f"Unchanged slots     : {len(comparison) - len(changed)}")

print("\nChanged slot details:")

if not changed:
    print("No predictions changed.")
else:
    for row in changed:
        print(
            f"{row['slot_id']}: "
            f"{row['original_prediction']} -> "
            f"{row['improved_prediction']} | "
            f"confidence: "
            f"{row['original_confidence'] * 100:.2f}% -> "
            f"{row['improved_confidence'] * 100:.2f}%"
        )

print(f"\nComparison CSV: {OUTPUT_CSV.resolve()}")
