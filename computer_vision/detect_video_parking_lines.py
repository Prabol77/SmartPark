
import cv2
import numpy as np
from pathlib import Path

INPUT = Path("runs/video_analysis/temporal_background_enhanced.jpg")
OUTPUT = Path("runs/video_analysis/parking_line_diagnostics.jpg")

image = cv2.imread(str(INPUT))

if image is None:
    raise FileNotFoundError(INPUT)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Restrict detection to the actual parking region.
roi_mask = np.zeros_like(gray)
roi_mask[110:440, 180:1270] = 255

# Detect bright painted markings.
white_mask = cv2.inRange(gray, 180, 255)
white_mask = cv2.bitwise_and(white_mask, roi_mask)

# Suppress small details and isolated vehicle edges.
white_mask = cv2.medianBlur(white_mask, 5)

edges = cv2.Canny(white_mask, 60, 160)

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=45,
    minLineLength=40,
    maxLineGap=8
)


# Reduce noise while preserving line structures.
edges = cv2.Canny(white_mask, 50, 150)

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=25,
    minLineLength=25,
    maxLineGap=18
)

result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

accepted = []

if lines is not None:
    for line in lines:
        values = np.asarray(line).reshape(-1)

        if values.size != 4:
            continue

        x1, y1, x2, y2 = map(int, values)

        dx = x2 - x1
        dy = y2 - y1

        angle = np.degrees(np.arctan2(dy, dx))
        length = np.hypot(dx, dy)

        if 20 <= abs(angle) <= 75 and length >= 25:
            accepted.append((x1, y1, x2, y2, angle, length))

            cv2.line(
                result,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        dx = x2 - x1
        dy = y2 - y1

        angle = np.degrees(np.arctan2(dy, dx))

        length = np.hypot(dx, dy)

        # Retain diagonal parking-line candidates.
        if 20 <= abs(angle) <= 75 and length >= 25:
            accepted.append((x1, y1, x2, y2, angle, length))

            cv2.line(
                result,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

print("Total detected segments:", 0 if lines is None else len(lines))
print("Diagonal candidates:", len(accepted))

cv2.imwrite(str(OUTPUT), result)
cv2.imwrite(
    str(OUTPUT.parent / "parking_line_edges.jpg"),
    edges
)

print("Saved:", OUTPUT)
