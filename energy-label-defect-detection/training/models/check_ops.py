from ultralytics import YOLO

model = YOLO("best.pt")

model.export(
    format="onnx",
    imgsz=320,
    dynamic=False,
    opset=17
)