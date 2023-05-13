import cv2
import numpy as np
import os,math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pprint
import random
import utils.utils as utils 
import utils.court as court 

for id in range(1, 801):
    id = str(id).zfill(5)
    video_filename = F'val/{id}/{id}.mp4'
    video = cv2.VideoCapture(video_filename)
    success,frame = video.read()
    a = court.get_court(frame)
    print(a)
    cv2.waitKey(0)
    