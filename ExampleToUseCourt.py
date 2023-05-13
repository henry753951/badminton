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
    a,c = court.get_court(frame)
    print(c)
    for p in a:
        cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
    for p in c:
        cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 255, 255), -1)
    cv2.imshow('frame',frame)
    cv2.waitKey(0)
    