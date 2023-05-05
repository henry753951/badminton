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
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # hsv_inrange = cv2.inRange(hsv, (0, 0, 50), (255, 80, 255))
        cv2.imshow('gray', frame)
        white_mask = np.zeros_like(frame)
        t1=6
        t2=10
        sl=128
        sd=20
        for x in range(t2, 1280-t2):
            for y in range(t2, 720-t2):
                isWhite = False
                here = ycrcb[y][x][0]
                if here > sl:
                    for t in range(t1, t2+1):
                        isWhite |= (here - ycrcb[y][x-t][0]>sd) and (here - ycrcb[y][x+t][0]>sd)
                        isWhite |= (here - ycrcb[y-t][x][0]>sd) and (here - ycrcb[y+t][x][0]>sd)
                if isWhite:
                    white_mask[y][x] = 255
        cv2.imshow('white_mask', white_mask)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        edges = cv2.Canny(thresh, 100, 150)
        
        
        lines = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10
        )



        h_lines, v_lines = segment_lines(lines, 280, 0.5)
        court = np.zeros_like(frame)
        line_groups = {'h': {}, 'v': {}}
        for line in lines:
            x1, y1, x2, y2 = line[0]
            k = (y2 - y1) / (x2 - x1)  # 斜率
            b = y1 - k*x1  # 截距
            if abs(k) < 0.5:  # 水平线
                if k not in line_groups['h']:
                    line_groups['h'][k] = []
                line_groups['h'][k].append((x1, y1, x2, y2, b))
            elif abs(k) > 5:  # 垂直线
                if x1 not in line_groups['v']:
                    line_groups['v'][x1] = []
                line_groups['v'][x1].append((x1, y1, x2, y2, b))

        # 对每个水平线分组中的直线进行连接
        connected_lines = {'h': [], 'v': []}
        for k, lines in line_groups['h'].items():
            lines = sorted(lines, key=lambda x: x[4])  # 根据截距排序
            connected = []
            for line in lines:
                if len(connected) == 0:
                    connected.append(line)
                else:
                    last_line = connected[-1]
                    if abs(line[1] - last_line[1]) < 10:  # 判断是否在同一水平线上
                        if line[0] > last_line[2]:  # 判断是否相邻
                            connected.append(line)
                        else:
                            connected[-1] = (last_line[0], last_line[1], line[2], line[3], last_line[4])
                    else:
                        connected_lines['h'].extend(connected)
                        connected = [line]
            connected_lines['h'].extend(connected)

        # 对每个垂直线分组中的直线进行连接
        for x, lines in line_groups['v'].items():
            lines = sorted(lines, key=lambda x: x[4])  # 根据截距排序
            connected = []
            for line in lines:
                if len(connected) == 0:
                    connected.append(line)
                else:
                    last_line = connected[-1]
                    if abs(line[0] - last_line[0]) < 10:  # 判断是否在同一垂直线上
                        if line[1] > last_line[3]:  # 判断是否相邻
                            connected.append(line)
                        else:
                            connected[-1] = (line[0], last_line[1], line[2], last_line[3], last_line[4])
                    else:
                        connected_lines['v'].extend(connected)
                        connected = [line]
            connected_lines['v'].extend(connected)

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