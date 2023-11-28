from ultralytics import YOLO
import os
from ray import tune
from ray.tune.examples.mnist_pytorch import train_mnist

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    # dataset_path = '/Users/saminator/Documents/yolov8/data/potato.yaml'
    dataset_path = 'c:/Users/USER/Documents/samin/yolov8/data/potato.yaml'
    # dataset_path = '/home/wakanda/Documents/samin/yolov8/data/potato.yaml'
    train_args = dict(epochs=500, batch=4, imgsz=640, project="potato", name="potato-", augment=True, visualize=True, device=0, optimizer="auto")
    train_args['data'] = dataset_path
    train_args['classes'] = [0, 23]
    train_args['fraction'] = 1.0
    train_args['patience'] = 30
    train_args['pretrained'] = True
    train_args['freeze'] = 1
    # train_args['iou'] = 0.5
    train_args['cos_lr'] = True
    augment_args = dict()
    augment_args['lr0']= 1E-3 # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    augment_args['lrf']= 0.1 # final learning rate (lr0 * lrf)
    augment_args['momentum']= 0.937 # SGD momentum/Adam beta1
    augment_args['weight_decay']= 0.0005 # optimizer weight decay 5e-4
    # augment_args['warmup_epochs']= 3.0 # warmup epochs (fractions ok)
    # augment_args['warmup_momentum']= 0.8 # warmup initial momentum
    # augment_args['warmup_bias_lr']= 0.1 # warmup initial bias lr
    # augment_args['box']= 0.05 # box loss gain
    # augment_args['cls']= 0.03 # cls loss gain (scale with pixels)
    # # augment_args['dfl']= 0.56016 # dfl loss gain
    # # augment_args['pose']= 12.0 # pose loss gain
    # # augment_args['kobj']= 1.0 # keypoint obj loss gain
    # # augment_args['label_smoothing']= 0.0 # label smoothing (fraction)
    # augment_args['nbs']= 64 # nominal batch size
    # augment_args['hsv_h']= 0.015 # image HSV-Hue augmentation (fraction)
    # augment_args['hsv_s']= 0.7 # image HSV-Saturation augmentation (fraction)
    # augment_args['hsv_v']= 0.4 # image HSV-Value augmentation (fraction)
    # augment_args['degrees']= 0.1 # image rotation (+/- deg)
    # augment_args['translate']= 0.1 # image translation (+/- fraction)
    # augment_args['scale']= 0 # image scale (+/- gain)
    augment_args['shear']= 0.1 # image shear (+/- deg)
    # augment_args['perspective']= 0.0 # image perspective (+/- fraction), range 0-0.001
    augment_args['flipud']= 0.5 # image flip up-down (probability)
    # augment_args['fliplr']= 0.5 # image flip left-right (probability)
    augment_args['mosaic']= 0.5 # image mosaic (probability)
    # augment_args['mixup']= 0.0 # image mixup (probability)
    augment_args['copy_paste']= 0.1 # segment copy-paste (probability)

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')
    # model = YOLO('yolov8s.yaml').load('yolov8s.pt')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    # results = model.tune(**train_args, **augment_args, use_ray=True, gpu_per_trial=1)
    results = model.train(**train_args, **augment_args)
    
    # Evaluate the model's performance on the validation set
    # results = model.val()

    # Tuning continued
    # experiment_path = os.path.join('C:/Users/USER/ray_results', '_tune_2023-11-18_08-00-07')
    # print(f"Loading results from {experiment_path}...")

    # restored_tuner = tune.Tuner.restore(experiment_path, trainable=train_mnist)
    # results = restored_tuner.get_results()

    # if results.errors:
    #     print("One or more trials failed!")
    # else:
    #     print("No errors!")

    # num_results = len(results)
    # print("Number of results:", num_results)

    # Iterate over results
    # for i, result in enumerate(results):
    #     if result.error:
    #         print(f"Trial #{i} had an error:", result.error)
    #         continue

    #     print(
    #         f"Trial #{i} finished successfully with a mean accuracy metric of:",
    #         result.metrics
    #     )
    # print(results)
    # results_df = results.get_dataframe().to_csv('results.csv')
    # results_df[["training_iteration", "mean_accuracy"]]

    # print("Shortest training time:", results_df["time_total_s"].min())
    # print("Longest training time:", results_df["time_total_s"].max())

    # best_result_df = results.get_dataframe(
    # filter_metric="mean_accuracy", filter_mode="max"
    # )
    # best_result_df[["training_iteration", "mean_accuracy"]]
    # from ray.train import Result

    # # Get the result with the maximum test set `mean_accuracy`
    # best_result: Result = results.get_best_result()

    # # Get the result with the minimum `mean_accuracy`
    # worst_performing_result: Result = results.get_best_result(
    #     metric="mean_accuracy", mode="min"
    # )
    # print(best_result.config)


    # Perform object detection on an image using the model
    # results = model('https://ultralytics.com/images/bus.jpg')

    # Export the model to ONNX format
    # success = model.export(format='onnx')
