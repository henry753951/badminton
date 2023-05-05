import cv2, math
import pandas as pd
import random, os
import numpy as np


def segment_lines(lines, deltaX, deltaY):
    h_lines = []
    v_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(y2 - y1) < deltaY:  # y-values are near; line is horizontal
                h_lines.append(line)
            elif abs(x2 - x1) < deltaX:  # x-values are near; line is vertical
                v_lines.append(line)
    return h_lines, v_lines



# 1 to 800
for id in range(1, 801):
    id = str(id).zfill(5)
    video_filename = F'../train/{id}/{id}.mp4'
    bestNW = 150,100
    bestNE = 350,100
    bestSW = 100,400
    bestSE = 400,400
    cap = cv2.VideoCapture(video_filename)
    border = np.zeros((720,1280,3), np.uint8)
    # 建立背景差分器

    ret, frame = cap.read()
    if ret:
        imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # 二值化
        ret,thresh = cv2.threshold(imgray,190,255,0)
        # 找輪廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = list(contours)
        # 根據輪廓面積大小進行sort
        contours.sort(key = cv2.contourArea , reverse=True)
        # 畫出前20的輪廓
        cv2.drawContours(border, contours[0:20], -1, (0,0,255), 10)

        # 每幀全部 -3
        cv2.subtract(border,(3,3,3,0) , border)

        cv2.imshow('frame' , border)

        imgray = cv2.cvtColor(border,cv2.COLOR_BGR2GRAY)
        # 變黑白
        ret,thresh = cv2.threshold(imgray,50,255,0)
        # 找邊緣
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 找最大的面積 = 球場
        c = max(contours, key = cv2.contourArea)
        # 找凸包
        hull = cv2.convexHull(c)
        epsilon = 0.01*cv2.arcLength(hull,True)
        approx = cv2.approxPolyDP(hull,epsilon,True)
        # 劃出球場
        clean_border = np.zeros((720,1280,3), np.uint8)

        cv2.drawContours(clean_border, [approx], -1, (0,255,255), 2)

        cv2.imshow('test', clean_border)
        # cv2.imshow("court", court)
        # cv2.imshow("court_", court_)
        # cv2.imshow("gray", gray)
        cv2.waitKey(0)
        
    cap.release()
cv2.destroyAllWindows()