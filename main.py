import torch  # YOLOv5 implemented using PyTorch
from IPython.display import Image, clear_output  # to display images
from tqdm import tqdm  # progress bar
import IProgress  # progress bar
from ipywidgets import IntProgress  # progress bar
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#torch.cuda.get_arch_list()


# Clone the YOLOv5 repository
# git clone https://github.com/ultralytics/yolov5
# %cd yolov5
# pip install -r requirements.txt


# Train the model
# python train.py --img 640 --epochs 10 --data L:/Programming/YOLOv5-Training/dataset/data.yaml --weights yolov5x.pt --device 0 --cache disk


# Test the YOLOv5 model on a single image
# python detect.py --weights runs/train/exp7/weights/best.pt --img 640 --source data/images/4049.jpg


# Convert weights to fp16 TFLite model
# python export.py --weights runs/train/exp7/weights/best.pt --include tflite --img 640


# Convert weights to fp16 TFLite model making sure to use the data.yaml file from the training.
# python export.py --data L:/Programming/YOLOv5-Training/dataset/data.yaml --weights L:/Programming/YOLOv5-Training/yolov5/runs/train/exp7/weights/best.pt --include tflite --img 640 --device 0


# Test the TFLite model
# python detect.py --weights runs/train/exp7/weights/best-fp16.tflite --img 640 --source data/images/4049.jpg