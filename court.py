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

    # 建立背景差分器

    ret, frame = cap.read()
    if ret:
        contrast = 30
        brightness = 20
        frame = frame * (contrast/127 + 1) - contrast + brightness # 轉換公式
        frame = np.clip(frame, 0, 255)
        frame = np.uint8(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        border = np.zeros_like(gray)
 
        
        edges = cv2.Canny(gray, 100, 150)
        dilated = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8))
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = list(contours)
        # 根據輪廓面積大小進行sort
        contours.sort(key = cv2.contourArea , reverse=True)
        # 畫出前20的輪廓
        cv2.drawContours(border, contours[0:10], -1, (255,255,255), 10)


        c = max(contours, key = cv2.contourArea)
        # 找凸包
        hull = cv2.convexHull(c)
        epsilon = 0.01*cv2.arcLength(hull,True)
        approx = cv2.approxPolyDP(hull,epsilon,True)

        cv2.drawContours(clean_border, [approx], -1, (0,255,255), 2)
        lines = cv2.HoughLinesP(
                border, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10
        )


        h_lines, v_lines = segment_lines(lines, 280, 0.5)


        court = np.zeros_like(frame)
        line_groups = {'h': {}, 'v': {}}
        for line in lines:
            x1, y1, x2, y2 = line[0]
            k = (y2 - y1) / (x2 - x1) 
            b = y1 - k*x1 
            if abs(k) < 0.5: 
                if k not in line_groups['h']:
                    line_groups['h'][k] = []
                line_groups['h'][k].append((x1, y1, x2, y2, b))
            elif abs(k) > 5: 
                if x1 not in line_groups['v']:
                    line_groups['v'][x1] = []
                line_groups['v'][x1].append((x1, y1, x2, y2, b))

        connected_lines = {'h': [], 'v': []}
        for k, lines in line_groups['h'].items():
            lines = sorted(lines, key=lambda x: x[4]) 
            connected = []
            for line in lines:
                if len(connected) == 0:
                    connected.append(line)
                else:
                    last_line = connected[-1]
                    if abs(line[1] - last_line[1]) < 5:
                        if line[0] > last_line[2]: 
                            connected.append(line)
                        else:
                            connected[-1] = (last_line[0], last_line[1], line[2], line[3], last_line[4])
                    else:
                        connected_lines['h'].extend(connected)
                        connected = [line]
            connected_lines['h'].extend(connected)

        #for x, lines in line_groups['v'].items():
        #    lines = sorted(lines, key=lambda x: x[4])
        #    connected = []
        #    for line in lines:
        #        if len(connected) == 0:
        #            connected.append(line)
        #        else:
        #            last_line = connected[-1]
        #            if abs(line[0] - last_line[0]) < 50: 
        #                if line[1] > last_line[3]: 
        #                    connected.append(line)
        #                else:
        #                    connected[-1] = (line[0], last_line[1], line[2], last_line[3], last_line[4])
        #            else:
        #                connected_lines['v'].extend(connected)
        #                connected = [line]
        #    connected_lines['v'].extend(connected)
        # v_lines = connected_lines['v']
        h_lines = connected_lines['h']

        # Drawing Horizontal Hough Lines on image
        for i in range(len(v_lines)):
            l = v_lines[i][0]
            cv2.line(court, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)

        # Drawing Horizontal Hough Lines on image
        for i in range(len(h_lines)):
            l = h_lines[i]
            cv2.line(court, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)



        # cv2.imshow('00', mask)
        cv2.imshow("court", court)
        # cv2.imshow("court_", court_)
        # cv2.imshow("gray", gray)
        cv2.waitKey(0)
        
    cap.release()
cv2.destroyAllWindows()