from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")  # fixed variable name

file_path = r"C:\Users\Fatema Kotb\Documents\CUFE 25\Year 04\GP\Spring\El7a2ny-Graduation-Project\CPR\Dataset\Crowds\vid1.mp4"  # fixed variable name

results = model(source=file_path, show=True, save=True, project="runs/detect", name="yolo11n-pose")