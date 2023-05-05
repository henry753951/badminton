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



    ret, frame = cap.read()
    if ret:

        contrast = 200
        brightness = 200
        frame = frame * (contrast/127 + 1) - contrast + brightness # 轉換公式
        frame = np.clip(frame, 0, 255)
        frame = np.uint8(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_inrange = cv2.inRange(hsv, (0, 0, 50), (255, 80, 255))
        
        thresh = cv2.threshold(hsv_inrange, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        edges = cv2.Canny(thresh, 100, 150)
        dilated = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8))
        
        lines = cv2.HoughLinesP(
                dilated, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10
        )



        h_lines, v_lines = segment_lines(lines, 280, 0.5)
        h_linesP = []   
        for h_line in h_lines:
            x1, y1, x2, y2 = h_line[0]
            max_x = x1
            min_x = x2
            for h_line_ in h_lines:
                x1_, y1_, x2_, y2_ = h_line_[0]
                if abs(x1_ - x1) < 50 and x1_ > max_x:
                    max_x = x1_
                if abs(x2_ - x2) < 50 and x2_ < min_x:
                    min_x = x2_
            h_linesP.append([max_x, y1, min_x, y2])

        court = np.zeros_like(thresh)
        court_ = np.zeros_like(thresh)

        # Drawing Horizontal Hough Lines on image
        for i in range(len(v_lines)):
            l = v_lines[i][0]
            cv2.line(court, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)

        # Drawing Horizontal Hough Lines on image
        for i in range(len(h_lines)):
            l = h_lines[i][0]
            cv2.line(court, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow("court", court)
        contours, _ = cv2.findContours(court, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = list(contours)
        # 根據輪廓面積大小進行sort
        contours.sort(key = cv2.contourArea , reverse=True)
        # 畫出前20的輪廓
        cv2.drawContours(court_, contours[0:15], -1, (255,0,255), 4)

        # 每幀全部 -3
        cv2.subtract(court_,(3,3,3,0) , court_)


        # cv2.imshow('00', mask)
        cv2.imshow("court", court)
        cv2.imshow("court_", court_)
        # cv2.imshow("gray", gray)
        cv2.waitKey(0)
        
    cap.release()
cv2.destroyAllWindows()