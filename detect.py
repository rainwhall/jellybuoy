from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("best.pt")

results = model(source=0, show=True, stream=True)
for result in results:
    boxes = result.boxes
    classes = result.names
