"""Run the currently installed parking-slot detector against a small image set.

Usage from project root:
    python diagnose_parking_images.py --images demo1.jpg demo2.jpg demo3.jpg demo4.jpg demo5.jpg

This is a diagnostic only. Candidate counts are not ground-truth accuracy.
"""
import argparse
import csv
from pathlib import Path
import cv2
from computer_vision.automatic_slot_detection import detect_slots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--output', default='runs/detector_diagnostics')
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    for image_name in args.images:
        image_path = Path(image_name)
        image = cv2.imread(str(image_path))
        if image is None:
            rows.append({'image': str(image_path), 'status': 'IMAGE_READ_FAILED', 'slots': 0, 'rows': 0})
            continue

        result = detect_slots(image)
        metadata = result[0] if isinstance(result, tuple) else result
        slots = metadata.get('slots', []) if isinstance(metadata, dict) else []
        row_reports = metadata.get('row_reports', metadata.get('row_bands', [])) if isinstance(metadata, dict) else []

        preview = image.copy()
        for slot in slots:
            points = slot.get('points')
            if points:
                polygon = cv2.UMat if False else None  # keep this script CPU-only
                import numpy as np
                pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(preview, [pts], True, (0, 220, 255), 2)
                center = np.mean(pts.reshape(-1, 2), axis=0).astype(int)
                cv2.putText(preview, slot.get('slot_id', '?'), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

        preview_path = output / f'{image_path.stem}_detector_preview.jpg'
        cv2.imwrite(str(preview_path), preview)
        rows.append({
            'image': str(image_path),
            'status': 'OK' if slots else 'NO_CANDIDATES',
            'slots': len(slots),
            'rows': len(row_reports),
            'preview': str(preview_path),
        })
        print(f"{image_path.name}: {len(slots)} candidate slots, {len(row_reports)} detected row groups")

    report = output / 'detector_diagnostic.csv'
    fields = sorted({key for row in rows for key in row})
    with report.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nDiagnostic report: {report.resolve()}')
    print('Important: a non-zero candidate count does not prove that the polygons match real parking bays.')


if __name__ == '__main__':
    main()
