from ultralytics import YOLO

model = YOLO("best.pt")

model.export(
    format="onnx",
    opset=13,
    simplify=True,
    dynamic=False
)