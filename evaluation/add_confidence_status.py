
import argparse
import csv
from collections import Counter
from pathlib import Path


def get_confidence_percent(value):
    confidence = float(value)

    if 0 <= confidence <= 1:
        confidence *= 100

    if not 0 <= confidence <= 100:
        raise ValueError(f"Invalid confidence value: {value}")

    return confidence


def classify_confidence(confidence):
    if confidence >= 95:
        return "HIGH_CONFIDENCE"
    elif confidence >= 80:
        return "REVIEW"
    return "UNCERTAIN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="runs/parking_inference/confidence/"
                "parking_sample_confidence_report.csv"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with input_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV has no header.")

        required = {"prediction", "confidence"}
        missing = required - set(reader.fieldnames)

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if not rows:
        raise ValueError("CSV contains no predictions.")

    for column in ("confidence_percent", "confidence_status"):
        if column not in fieldnames:
            fieldnames.append(column)

    status_counts = Counter()
    class_counts = Counter()
    grouped_counts = Counter()
    confidence_values = []

    for row in rows:
        confidence = get_confidence_percent(row["confidence"])
        status = classify_confidence(confidence)
        prediction = row["prediction"].strip().upper()

        row["confidence_percent"] = f"{confidence:.2f}"
        row["confidence_status"] = status

        status_counts[status] += 1
        class_counts[prediction] += 1
        grouped_counts[(prediction, status)] += 1
        confidence_values.append(confidence)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== CONFIDENCE REPORT ==========")
    print(f"Total predictions: {len(rows)}")
    print(f"Mean confidence: {sum(confidence_values) / len(rows):.2f}%")

    print("\nConfidence categories:")
    for status in ("HIGH_CONFIDENCE", "REVIEW", "UNCERTAIN"):
        count = status_counts[status]
        print(f"{status}: {count} ({count / len(rows) * 100:.2f}%)")

    print("\nPredicted classes:")
    for prediction, count in sorted(class_counts.items()):
        print(f"{prediction}: {count}")

    print("\nClass-wise confidence review:")
    for (prediction, status), count in sorted(grouped_counts.items()):
        print(f"{prediction} | {status}: {count}")

    print(f"\nReport saved to: {output_path}")
    print("Note: Confidence is not verified prediction accuracy.")


if __name__ == "__main__":
    main()
