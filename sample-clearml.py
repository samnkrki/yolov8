from clearml import Task
from ultralytics import YOLO 

# Create a ClearML Task
task = Task.init(
    project_name="my project",
    task_name="my yolo task"
)

# Load a model
model_variant = "yolov8n"
# Log "model_variant" parameter to task
task.set_parameter("model_variant", model_variant)

# Load the YOLOv8 model
model = YOLO(f'{model_variant}.pt') 

# Put all YOLOv8 arguments in a dictionary and pass it to ClearML
# When the arguments are later changed in UI, they will be overridden here!
args = dict(data="coco128.yaml", epochs=3)
task.connect(args)

# Train the model 
# If running remotely, the arguments may be overridden by ClearML if they were changed in the UI
results = model.train(**args)