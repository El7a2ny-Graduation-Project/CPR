from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")  # fixed variable name

results = model(source="real.mp4", show=True, save=True, project="runs/detect", name="yolo11n-pose")