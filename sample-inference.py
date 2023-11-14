from PIL import Image
import os
from ultralytics import YOLO
import shutil
# Define output folder
output_folder = 'out'

def save_inference(input_folder, model):
    # Loop over all image files in input directory
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load image and run inference
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            results = model(img, save_txt=True)
            print(results)
            # Save detection output to runs/detect/ folder
            # results.save()

            # # Move inference output to specified output folder
            # timestamp = os.listdir('runs/detect/')[0]
            # shutil.move(os.path.join('runs/detect', timestamp), os.path.join(output_folder, timestamp))

if __name__ == '__main__':
    dataset_path = 'C:/Users/USER/Documents/datasets/potato_many/train_2560/images'
    model = YOLO('./potato/potato-27/weights/best.pt')
    save_inference(dataset_path, model)