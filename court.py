import cv2, math
import pandas as pd
import random, os
import numpy as np

def get_intersections(line1, line2):
    A = np.array(line1)
    B = np.array(line2)
    t, s = np.linalg.solve(np.array([A[1]-A[0], B[0]-B[1]]).T, B[0]-A[0])    
    return (1-t)*A[0] + t*A[1]

def getPointLineDistance(x, y, line):
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    dist = abs(dy * x - dx * y + x2 * y1 - y2 * x1) / math.sqrt(dx ** 2 + dy ** 2)
    # 計算點與線的距離，並判斷正負
    if dy * x < dx * y - x2 * y1 + y2 * x1:
        dist *= -1
    return dist

def get_angle(line1, line2):
    d1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1])
    d2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1])
    p = d1[0] * d2[0] + d1[1] * d2[1]
    n1 = math.sqrt(d1[0] * d1[0] + d1[1] * d1[1])
    n2 = math.sqrt(d2[0] * d2[0] + d2[1] * d2[1])
    ang = math.acos(p / (n1 * n2))
    ang = math.degrees(ang)
    return ang

def getPerspectiveTransformMatrix(frame , show_frame = False):
    border = np.zeros((720,1280,3), np.uint8)


    test_frame =np.copy(frame)

    h, w = frame.shape[:2]
    mask = np.zeros([h+2, w+2], np.uint8) 
    # bgr
    diff_value = (3,1,3)
    black = (0,0,0)
    frame = cv2.GaussianBlur(frame , (7,7) , 0)

    fillpoints = [(100,0) , (100,700) , (300,350) , (1000,350) , (1200,700)]
    
    for point in fillpoints:
        cv2.floodFill(frame, mask, point, black, diff_value, diff_value)
    for point in fillpoints:
        cv2.circle(frame , point , 5 ,(0,0,255) , -1)

    # 灰階
    imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # 二值化
    ret,thresh = cv2.threshold(imgray,100,255,0)
    # 找輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = list(contours)
    # 根據輪廓面積大小進行sort
    contours.sort(key = cv2.contourArea , reverse=True)
    # 畫出最大的輪廓
    cv2.drawContours(border, contours[0:1], -1, (0,0,255), 10)



    # 灰階
    imgray = cv2.cvtColor(border,cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(imgray, 30, 100)
    lines = cv2.HoughLinesP(canny , 1.0 , np.pi/180 , 100 , np.array([]) , 200 , 100)

    cv2.line(frame,(100,100),(100,500),(255,255,255),5)

    horizon_line = []
    left_vertical_line = None
    right_vertical_line = None

    try:
    # 畫球場的垂直線 和 找到水平線座標
        for line in lines:
            for x1,y1,x2,y2 in line:
                line_angle = get_angle([(x1,y1),(x2,y2)] , [(0,0),(0,100)])
                line_angle_90 = 180 - line_angle if line_angle > 90 else line_angle
                vectorx = x2 - x1
                vectory = y2 - y1
                
                if  (line_angle_90 < 40 and int(line_angle)): 

                    if left_vertical_line == None and line_angle > 90:
                        # left_vertical_line = [(x1 - 20 , y1) , (x2 -20, y2)]
                        left_vertical_line = [(x1 , y1) , (x2, y2)]
                        cv2.line(frame,(x1 - vectorx * 100, y1 - vectory * 100),(x1 + vectorx * 100,y1 + vectory * 100),(255,0,0),5)
                    elif right_vertical_line == None and line_angle < 90:
                        right_vertical_line = [(x1, y1) , (x2, y2)]
                        cv2.line(frame,(x1 - vectorx * 100, y1 - vectory * 100),(x1 + vectorx * 100,y1 + vectory * 100),(255,0,0),5)

                elif line_angle_90 > 85:

                    horizon_line.append([[x1 ,y1] , [x2,y2] ])

        # 畫上下兩條水平線
        top_line = min(horizon_line , key = lambda x : x[0][1] + x[1][1])
        # top_line[0][1] -= 20
        # top_line[1][1] -= 20
        x1 , y1 = top_line[0]
        x2 , y2 = top_line[1]
        cv2.line(frame,(x1 - (x2-x1) * 100, y1 - (y2-y1) * 100),(x1 + (x2-x1) * 100,y1 + (y2-y1) * 100),(255,0,0),5)    
        bottom_line = max(horizon_line , key = lambda x : x[0][1] + x[1][1])
        x1 , y1 = bottom_line[0]
        x2 , y2 = bottom_line[1]
        cv2.line(frame,(x1 - (x2-x1) * 100, y1 - (y2-y1) * 100),(x1 + (x2-x1) * 100,y1 + (y2-y1) * 100),(255,0,0),5)    

        # print(get_intersections(top_line , vertical_line[0]).astype(int))
        corner = []
        corner.append(get_intersections(top_line , left_vertical_line).astype(int))
        corner.append(get_intersections(bottom_line , left_vertical_line).astype(int))
        corner.append(get_intersections(bottom_line , right_vertical_line).astype(int))
        corner.append(get_intersections(top_line , right_vertical_line).astype(int))
        cv2.circle(frame , get_intersections(top_line , left_vertical_line).astype(int) , 5 , (0,255,0),-1)
        cv2.circle(frame , get_intersections(top_line , right_vertical_line).astype(int) , 5 , (0,255,0),-1)
        cv2.circle(frame , get_intersections(bottom_line , left_vertical_line).astype(int) , 5 , (0,255,0),-1)
        cv2.circle(frame , get_intersections(bottom_line , right_vertical_line).astype(int) , 5 , (0,255,0),-1)
        
    except:
        return []
        

    # # cv2.imshow('board' , border)          
            
    # # 進行透視變換
    # old = np.float32(corner)
    # new = np.float32([[0,0], [0,h-1], [w-1,h-1] , [w-1,0] ])
    # matrix = cv2.getPerspectiveTransform(old , new)
    # imgOutput = cv2.warpPerspective(test_frame, matrix, (w , h), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    return corner



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
        try:
            cv2.destroyWindow("court_mask")
        except:
            pass
        corner = getPerspectiveTransformMatrix(frame,  show_frame = False)
        if len(corner) == 4:
            court_mask = np.zeros_like(frame)
            cv2.fillPoly(court_mask, np.array([corner], dtype=np.int32), (255, 255, 255))
            # cv2.imshow("court_mask", court_mask)
            frame = cv2.bitwise_and(frame, court_mask)
            

        contrast = -15
        brightness = -10
        frame = frame * (contrast/127 + 1) - contrast + brightness # 轉換公式
        frame = np.clip(frame, 0, 255)
        frame = np.uint8(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        blur = cv2.GaussianBlur(frame, (11,11), 0)
        usm = cv2.addWeighted(frame, 1.5, blur, -0.5, 0)

        gray = cv2.cvtColor(usm, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray,30,25,75)
        gray = cv2.GaussianBlur(gray, (17,17), 0)
        cv2.imshow("gray", gray)

        border = np.zeros_like(gray)
        court = np.zeros_like(frame)

        edges = cv2.Canny(gray, 100,300,apertureSize=5)
        edges = cv2.blur(edges, (11,11))  
        cv2.imshow("edges", edges)

        dst = cv2.cornerHarris(edges, 5, 9, 0.04)
        dst = cv2.dilate(dst, None)
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)


        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
        filter_corners = []
        # Drawing a circle around corners
        for i in range(1, len(corners)):
            point_hsvs = hsv[int(corners[i, 1])-5:int(corners[i, 1])+5:, int(corners[i, 0])-5: int(corners[i, 0])+5]
            for point_hsv_xs in point_hsvs:
                for point_hsv in point_hsv_xs:
                    if point_hsv[1]<30 and point_hsv[2]>60:
                        # cv2.circle(frame, (int(corners[i, 0]), int(corners[i, 1])), 5, (0, 0, 255), 2)
                        filter_corners.append(corners[i])
                        break
                    # else:
                    #     cv2.circle(frame, (int(corners[i, 0]), int(corners[i, 1])), 5, (0, 255, 0), 2)
                else:
                    continue
                break


            

        h_lines = []
        max_length_of_line = 0
        max_y = 0
        for i in range(len(filter_corners)):
            for j in range(i+1, len(filter_corners)):
                x1, y1 = filter_corners[i]
                x2, y2 = filter_corners[j]
                #  detact the line is horizontal
                if abs(y1 - y2) < 5:
                    if pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5)>max_length_of_line:
                        max_length_of_line = pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5)
                        max_y = y1
                    h_lines.append([x1, y1, x2, y2])
                    # cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        h_lines.sort(key=lambda x: pow(pow(x[0] - x[2], 2) + pow(x[1] - x[3], 2), 0.5), reverse=True)
        temp=[]
        
        for point in filter_corners:
            if point[1] <= max_y+5:
                temp.append(point)
        filter_corners = temp


        for p in filter_corners:
            # check p is on h_lines_f
            for line in h_lines[0:1]:
                x1, y1, x2, y2 = line
                cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 255, 0), 2)
                if y1-5 <= p[1] <= y2+5:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 0, 255), 2)
                    break




        # lines = cv2.HoughLinesP(
        #         edges, rho=1 , theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10
        # )

        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)


        lines = cv2.HoughLines(edges,1,np.pi/180,150)
        for line in lines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 2000*(-b))
                y1 = int(y0 + 2000*(a))
                x2 = int(x0 - 2000*(-b))
                y2 = int(y0 - 2000*(a))
                cv2.line(court,(x1,y1),(x2,y2),(0,0,255),2)

        
        cv2.imshow("court", court)
        cv2.imshow("frame", frame)
        
        cv2.waitKey(0)
        
    cap.release()
cv2.destroyAllWindows()