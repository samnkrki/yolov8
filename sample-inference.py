from PIL import Image
import os
from ultralytics import YOLO
import shutil
# Define output folder
output_folder = 'inference'

def save_inference(input_folder, model, project, name):
    # Loop over all image files in input directory
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load image and run inference
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            results = model(img, save_txt=True, project=project,exist_ok=True, name=name)
            print(results)
            # Save detection output to runs/detect/ folder
            # results.save()

            # # Move inference output to specified output folder
            # timestamp = os.listdir('runs/detect/')[0]
            # shutil.move(os.path.join('runs/detect', timestamp), os.path.join(output_folder, timestamp))

if __name__ == '__main__':
    project = 'potato'
    folder_name='train_2720'
    name="potato-49"
    path = f'./{project}/{name}/weights/best.pt'
    dataset_path = f'c:/Users/USER/Documents/datasets/potato_many/{folder_name}/images'
    model = YOLO(path)
    save_inference(dataset_path, model, project, name+'/'+folder_name)