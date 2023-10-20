from ultralytics import YOLO
from ray import tune

model = YOLO("yolov8n.pt")
results = model.tune(data="coco128.yaml", epochs=3, use_ray=True);

print(results)