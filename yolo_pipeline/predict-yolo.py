from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load a trained model
model = YOLO("invasion/train5/weights/best.pt")  # update path if needed

# Perform inference
results = model.val(data="my_data.yaml", imgsz=256, save=True, conf=0.45, iou=.5)  # or use a list of images
print(results)

