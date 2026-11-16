
import cv2
import torch
from ultralytics import YOLO

IMAGE_PATH = "datasets/public/parking_sample.jpg"

print("========== ENVIRONMENT ==========")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

print("Image dimensions:", image.shape)

print("\n========== LOADING MODEL ==========")

model = YOLO("yolo11n.pt")

print("Model classes:")
print(model.names)

print("\n========== RUNNING UNFILTERED DETECTION ==========")

results = model.predict(
    source=IMAGE_PATH,
    device=0,
    imgsz=1280,
    conf=0.001,
    iou=0.7,
    verbose=True
)

result = results[0]

print("\n========== DIAGNOSTIC RESULTS ==========")
print("Total raw detections:", len(result.boxes))

if len(result.boxes) > 0:
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        print(
            "Class:", model.names[class_id],
            "| Confidence:", round(confidence, 4)
        )
else:
    print("No objects detected at all.")

annotated = result.plot()

cv2.imwrite(
    "runs/vehicle_detection/diagnostic.jpg",
    annotated
)

print("\nDiagnostic image saved.")
