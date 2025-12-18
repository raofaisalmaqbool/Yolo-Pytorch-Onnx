from ultralytics import YOLO

# Load model
model = YOLO("model/yolo11n.pt")

# Export to ONNX
model.export(format="onnx", dynamic=False, simplify=True, imgsz=640)