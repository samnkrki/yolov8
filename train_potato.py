from ultralytics import YOLO
import os
# from ray import tune

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    # dataset_path = '/Users/saminator/Documents/yolov8/data/potato.yaml'
    dataset_path = 'c:/Users/USER/Documents/samin/yolov8/data/potato.yaml'
    # dataset_path = '/home/wakanda/Documents/samin/yolov8/data/potato.yaml'
    train_args = dict(epochs=100, batch=4, imgsz=640, project="potato", name="potato-", augment=True, visualize=True, device=0, optimizer="Adam")
    train_args['data'] = dataset_path
    train_args['classes'] = [0]
    augment_args = dict()
    augment_args['lr0']= 0.08591211666330036 # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    augment_args['lrf']= 0.678455581881708 # final learning rate (lr0 * lrf)
    augment_args['momentum']= 0.8484255646838734 # SGD momentum/Adam beta1
    augment_args['weight_decay']= 0.0008491207411328716 # optimizer weight decay 5e-4
    augment_args['warmup_epochs']= 0.7338494867123063 # warmup epochs (fractions ok)
    augment_args['warmup_momentum']= 0.38158103445764946 # warmup initial momentum
    # augment_args['warmup_bias_lr']= 0.1 # warmup initial bias lr
    augment_args['box']= 0.043838900664838525 # box loss gain
    augment_args['cls']= 3.978326462737403 # cls loss gain (scale with pixels)
    # # augment_args['dfl']= 0.56016 # dfl loss gain
    # # augment_args['pose']= 12.0 # pose loss gain
    # # augment_args['kobj']= 1.0 # keypoint obj loss gain
    # # augment_args['label_smoothing']= 0.0 # label smoothing (fraction)
    # augment_args['nbs']= 64 # nominal batch size
    augment_args['hsv_h']= 4.2636002823248643e-05 # image HSV-Hue augmentation (fraction)
    augment_args['hsv_s']= 0.5959737499224373 # image HSV-Saturation augmentation (fraction)
    augment_args['hsv_v']= 0.11557579376895563 # image HSV-Value augmentation (fraction)
    augment_args['degrees']= 18.24923038358967 # image rotation (+/- deg)
    augment_args['translate']= 0.24858666563876924 # image translation (+/- fraction)
    augment_args['scale']= 0.47681002138525624 # image scale (+/- gain)
    augment_args['shear']= 8.382338793487836 # image shear (+/- deg)
    augment_args['perspective']= 0.0001243827075651327 # image perspective (+/- fraction), range 0-0.001
    augment_args['flipud']= 0.1862331297252371 # image flip up-down (probability)
    augment_args['fliplr']= 0.28806696189630443 # image flip left-right (probability)
    augment_args['mosaic']= 0.5805690772400773 # image mosaic (probability)
    augment_args['mixup']= 0.619121254748873 # image mixup (probability)
    augment_args['copy_paste']= 0.8733866793454359 # segment copy-paste (probability)

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    # results = model.tune(**train_args, **augment_args, use_ray=True, gpu_per_trial=1)
    results = model.train(**train_args, **augment_args)
    
    # Evaluate the model's performance on the validation set
    results = model.val()

    # Perform object detection on an image using the model
    # results = model('https://ultralytics.com/images/bus.jpg')

    # Export the model to ONNX format
    # success = model.export(format='onnx')
