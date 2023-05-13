# import sys
# import getopt
# import numpy as np
# import os
# from glob import glob
# import piexif
# from keras.preprocessing.image import ImageDataGenerator
# from keras.utils.image_utils import img_to_array,array_to_img,load_img
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from keras.models import *
# from keras.layers import *
# import keras.backend as K
# from keras import optimizers
# import tensorflow as tf
# import cv2,math
# from os.path import isfile, join
# from PIL import Image
# import time
# BATCH_SIZE=1

import os
import cv2
import math
import torch
import parse
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageSequence


HEIGHT = 288
WIDTH = 512
# mag=1

##################################  Prediction Functions ##################################
def get_model(model_name, num_frame, input_type):
    """ Create model by name and the configuration parameter.

        args:
            model_name - A str of model name
            num_frame - An int specifying the length of a single input sequence
            input_type - A str specifying input type
                '2d' for stacking all the frames at RGB channel dimesion result in shape (H, W, F*3)
                '3d' for stacking all the frames at extra dimesion result in shape (F, H, W, 3)

        returns:
            model - A keras.Model
            input_shape - A tuple specifying the input shape (for model.summary)
    """
    # Import model
    if model_name == 'TrackNetV2':
        from models.model import TrackNetV2 as TrackNet

    if model_name in ['TrackNetV2']:
        model = TrackNet(in_dim=num_frame*3, out_dim=num_frame)
    
    return model

def get_pred_type(cx_pred, cy_pred, cx, cy, tolerance):
    """ Get the result type of the prediction.

        args:
            cx_pred, cy_pred - ints specifying the predicted coordinates
            cx, cy - ints specifying the ground-truth coordinates
            tolerance - A int speicfying the tolerance for FP1

        returns:
            A str specifying the result type of the prediction
    """
    pred_has_ball = False if (cx_pred == 0 and cy_pred == 0) else True
    gt_has_ball = False if (cx == 0 and cy == 0) else True
    if  not pred_has_ball and not gt_has_ball:
        return 'TN'
    elif pred_has_ball and not gt_has_ball:
        return 'FP2'
    elif not pred_has_ball and gt_has_ball:
        return 'FN'
    else:
        dist = math.sqrt(pow(cx_pred-cx, 2)+pow(cy_pred-cy, 2))
        if dist > tolerance:
            return 'FP1'
        else:
            return 'TP'
def get_frame_unit(frame_list, num_frame):
    """ Sample frames from the video.

        args:
            frame_list - A str of video file path with format '{data_dir}/{split}/match{match_id}/video/{rally_id}.mp4

        return:
            frames - A tf.Tensor of a mini batch input sequence
    """
    batch = []
    # Get the resize scaler
    h, w, _ = frame_list[0].shape
    h_ratio = h / HEIGHT
    w_ratio = w / WIDTH
    
    def get_unit(frame_list):
        """ Generate an input sequence from frame pathes and labels.

            args:
                frame_list - A numpy.ndarray of single frame sequence with shape (F,)

            returns:
                frames - A numpy.ndarray of resized frames with shape (H, W, 3*F)
        """
        frames = np.array([]).reshape(0, HEIGHT, WIDTH)

        # Process each frame in the sequence
        for img in frame_list:
            img = cv2.resize(img, (WIDTH, HEIGHT))
            img = np.moveaxis(img, -1, 0)
            frames = np.concatenate((frames, img), axis=0)
        
        return frames
    
    # Form a mini batch of input sequence
    for i in range(0, len(frame_list), num_frame):
        frames = get_unit(frame_list[i: i+num_frame])
        frames /= 255.
        batch.append(frames)

    batch = np.array(batch)
    return torch.FloatTensor(batch)

def predict(model,frame_queue,num_frame):
    x = get_frame_unit(frame_queue, num_frame)

    # Inference
    with torch.no_grad():
        y_pred = model(x.cuda())
        
    
    y_pred = y_pred.detach().cpu().numpy()
    h_pred = y_pred > 0.5
    h_pred = h_pred * 255.
    h_pred = h_pred.astype('uint8')
    h_pred = h_pred.reshape(-1, HEIGHT, WIDTH)
    return h_pred

# def predict(model,image1,image2,image3):
# 	unit = []
# 	#Adjust BGR format (cv2) to RGB format (PIL)
# 	x1 = image1[...,::-1]
# 	x2 = image2[...,::-1]
# 	x3 = image3[...,::-1]
# 	#Convert np arrays to PIL images
# 	x1 = array_to_img(x1)
# 	x2 = array_to_img(x2)
# 	x3 = array_to_img(x3)
# 	#Resize the images
# 	x1 = x1.resize(size = (WIDTH, HEIGHT))
# 	x2 = x2.resize(size = (WIDTH, HEIGHT))
# 	x3 = x3.resize(size = (WIDTH, HEIGHT))
# 	#Convert images to np arrays and adjust to channels first
# 	x1 = np.moveaxis(img_to_array(x1), -1, 0)
# 	x2 = np.moveaxis(img_to_array(x2), -1, 0)
# 	x3 = np.moveaxis(img_to_array(x3), -1, 0)
# 	#Create data
# 	unit.append(x1[0])
# 	unit.append(x1[1])
# 	unit.append(x1[2])
# 	unit.append(x2[0])
# 	unit.append(x2[1])
# 	unit.append(x2[2])
# 	unit.append(x3[0])
# 	unit.append(x3[1])
# 	unit.append(x3[2])
# 	unit=np.asarray(unit)	
# 	unit = unit.reshape((1, 9, HEIGHT, WIDTH))
# 	unit = unit.astype('float32')
# 	unit /= 255
# 	y_pred = model.predict(unit, batch_size=BATCH_SIZE)
# 	y_pred = y_pred > 0.5
# 	y_pred = y_pred.astype('float32')
# 	h_pred = y_pred[0]*255
# 	h_pred = h_pred.astype('uint8')
# 	return h_pred



