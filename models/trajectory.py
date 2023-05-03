import sys
import getopt
import numpy as np
import os
from glob import glob
import piexif
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import img_to_array,array_to_img,load_img
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
import keras.backend as K
from keras import optimizers
import tensorflow as tf
import cv2
from os.path import isfile, join
from PIL import Image
import time
BATCH_SIZE=1
HEIGHT=288
WIDTH=512
mag=1

def predict(model,image1,image2,image3):
	unit = []
	#Adjust BGR format (cv2) to RGB format (PIL)
	x1 = image1[...,::-1]
	x2 = image2[...,::-1]
	x3 = image3[...,::-1]
	#Convert np arrays to PIL images
	x1 = array_to_img(x1)
	x2 = array_to_img(x2)
	x3 = array_to_img(x3)
	#Resize the images
	x1 = x1.resize(size = (WIDTH, HEIGHT))
	x2 = x2.resize(size = (WIDTH, HEIGHT))
	x3 = x3.resize(size = (WIDTH, HEIGHT))
	#Convert images to np arrays and adjust to channels first
	x1 = np.moveaxis(img_to_array(x1), -1, 0)
	x2 = np.moveaxis(img_to_array(x2), -1, 0)
	x3 = np.moveaxis(img_to_array(x3), -1, 0)
	#Create data
	unit.append(x1[0])
	unit.append(x1[1])
	unit.append(x1[2])
	unit.append(x2[0])
	unit.append(x2[1])
	unit.append(x2[2])
	unit.append(x3[0])
	unit.append(x3[1])
	unit.append(x3[2])
	unit=np.asarray(unit)	
	unit = unit.reshape((1, 9, HEIGHT, WIDTH))
	unit = unit.astype('float32')
	unit /= 255
	y_pred = model.predict(unit, batch_size=BATCH_SIZE)
	y_pred = y_pred > 0.5
	y_pred = y_pred.astype('float32')
	h_pred = y_pred[0]*255
	h_pred = h_pred.astype('uint8')
	return h_pred

