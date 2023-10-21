from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    augment_args = {}
    training_args = {
        'epochs': 100,
        'patience': 20,
        'optimizer': 'adam',
    }

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data='data/potato.yaml', epochs=100, patience=20, batch=4, imgsz=640, project="potato", name="potato-", classes=[0], augment=True, visualize=True, device=0, optimizer="Adam", lr0=0.001)

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Perform object detection on an image using the model
    results = model('https://ultralytics.com/images/bus.jpg')

    # Export the model to ONNX format
    success = model.export(format='onnx')
