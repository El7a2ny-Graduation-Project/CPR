from ultralytics import YOLO

model = YOLO(r"C:\Users\Fatema Kotb\Documents\CUFE 25\Year 04\GP\Spring\pytorch_onnx_tensorflow\yolo11n-pose_flex.tflite") 

results = model(source="real.mp4", show=True, save=True, project="runs/detect", name="yolov11n-pose-tf")