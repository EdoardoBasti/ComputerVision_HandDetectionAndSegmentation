# -*- coding: utf-8 -*-
"""YOLOv5

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ri5cfKOB1FqA0joa6UWarrDDe_s6qiD1

# **Custom YOLOv5**
#### **Stefano Binotto** (*matr*. 2052421) with the special contribution of **Edoardo Bastianello** (*matr*. 2053077) for the shooting of the images to include in the datasets on which we trained and validated the model.

The labeling was performed using [Make Sense](https://www.makesense.ai/).

## 1. Setup

Clone the official repository, install all the necessary dependencies and load the datasets.
"""

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/ultralytics/yolov5  # clone

# install all the dependencies
# %cd yolov5 # move to the directory containing "requirements.txt"
# %pip install -qr requirements.txt  # install

# importing necessary modules
import torch
import utils

display = utils.notebook_init()  # checks

"""Load and unzip all the datasets."""

# mount my personal Google Drive "drive" folder 
# in order to speed up the loading of the images 
from google.colab import drive
drive.mount("/content/drive/")

# unzip trainingset and testset
!unzip -q ../drive/MyDrive/train_data.zip -d ../
!unzip -q ../drive/MyDrive/test.zip -d ../

"""## 2. Train on Custom Dataset

Training the model on the dataset I uploaded on my Drive folder. 

In order to perform transfer learning I had to rewrite the configuration file "***custom_data.yaml***" to specify the location of the training set and the classes we want the model to be able to predict.
The best model is saved in "runs/train/exp/weights/best.pt", while the model of the last epoch is saved in "runs/train/exp/weights/last.pt".

The pre-trained weigths we used as initialization are the one related to the [YOLOV5m.pt](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results#:~:text=for%20background%20images.-,Model%20Selection,-Larger%20models%20like) model, which has about 21 million parameters, which makes it a very accurate model but still quite light.

- Custom Trainingset: [link](https://drive.google.com/file/d/1B4DFxi3NhfrCJVFZX9WnrT5qizKkck9K/view?usp=sharing)
"""

# Fine tune YOLOv5s on custom dataset for 60 epochs, 5 mini-batch size
!python train.py --img 640 --batch 5 --epochs 60 --data custom_data.yaml --weights yolov5m.pt --cache

"""## 3. Validation
Validate our best model on the validation set I uploaded on my Drive folder. The model used is "runs/train/exp/weights/best.pt", which is the best one found during the training. "***custom_coco_testing.yaml***" is the configuration file I wrote in order to perform the validation on the custom dataset. I specified which dataset to use and the classes we want the model to be able to predict.
"""

# Run val.py on validationset
!python val.py --weights runs/train/exp/weights/best.pt --data custom_coco_testing.yaml --img 640 --iou 0.65 --half

"""## 4. Inference for single image test

Using the best weights achieved so far (the path to the weights in my case was: runs/train/exp/weights/best.pt), the script `detect.py`can run the model inference on a wide range of sources, and save results to the `runs/detect` folder. Example inference sources are:

```shell
python detect.py --source 0  # webcam
                          img.jpg  # image 
                          vid.mp4  # video
                          path/  # directory
                          path/*.jpg  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```
"""

# change "conf" value for different FP tolerance
!python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source ../test/images/

# to use during the test to display the predicted images 
display.Image(filename='runs/detect/exp28/01.jpg', width=600)

"""## 5. Export best model"""

#export the best model by converting the best weights "best.pt" into an .onnx file
!python export.py --weights runs/train/exp/weights/best.pt --include onnx