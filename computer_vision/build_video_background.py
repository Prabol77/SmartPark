
import cv2
import numpy as np
from pathlib import Path


VIDEO_PATH = Path("datasets/public/parking_video.mp4")
OUTPUT_DIR = Path("runs/video_analysis")
FRAME_DIR = OUTPUT_DIR / "background_samples"

SAMPLE_INTERVAL = 1.0  # One frame per second


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = total_frames / fps

    print(f"Video duration: {duration:.2f} seconds")
    print(f"FPS: {fps}")

    samples = []

    for timestamp in np.arange(0, duration, SAMPLE_INTERVAL):
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

        success, frame = cap.read()

        if not success:
            continue

        # Save individual sample for inspection.
        sample_path = FRAME_DIR / f"sample_{int(timestamp):03d}.jpg"
        cv2.imwrite(str(sample_path), frame)

        # Resize consistently before calculating the median.
        samples.append(frame)

        print(f"Collected frame at {timestamp:.0f}s")

    cap.release()

    if len(samples) < 5:
        raise RuntimeError("Not enough frames to construct background.")

    print(f"\nTotal samples: {len(samples)}")
    print("Building temporal median background...")

    # Pixel-wise median suppresses moving objects when they
    # occupy a pixel in fewer than half of the sampled frames.
    stack = np.stack(samples, axis=0)

    background = np.median(
        stack.astype(np.float32),
        axis=0
    ).astype(np.uint8)

    output_path = OUTPUT_DIR / "temporal_median_background.jpg"

    cv2.imwrite(
        str(output_path),
        background,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    # Also create a grayscale and enhanced version for line detection.
    gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    enhanced = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    cv2.imwrite(
        str(OUTPUT_DIR / "temporal_background_enhanced.jpg"),
        enhanced
    )

    print("\nBackground generation complete.")
    print(f"Background: {output_path}")
    print(f"Enhanced: {OUTPUT_DIR / 'temporal_background_enhanced.jpg'}")


if __name__ == "__main__":
    main()
