import cv2, math
import pandas as pd
import random, os
import numpy as np

def filter_lines(lines, k_values, n):
    top_k_values = sorted(k_values, reverse=True)[:n]
    filtered_lines = []
    for i in range(len(lines)):
        if k_values[i] in top_k_values:
            filtered_lines.append(lines[i])
    return filtered_lines

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

def findColorMost(frame):
    data = np.reshape(frame, (-1,3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans(data,1,None,criteria,10,flags)
    return centers[0].astype(np.int32)

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
for id in range(16, 801):
    id = str(id).zfill(5)
    video_filename = F'../train/{id}/{id}.mp4'
    bestNW = 150,100
    bestNE = 350,100
    bestSW = 100,400
    bestSE = 400,400
    cap = cv2.VideoCapture(video_filename)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
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
        else:
            court_mask = np.zeros_like(frame)
            
        # cv2.imshow("court_mask", court_mask)
        contrast = -15
        brightness = 15
        frame = frame * (contrast/127 + 1) - contrast + brightness # 轉換公式
        frame = np.clip(frame, 0, 255)
        frame = np.uint8(frame)
        bgrMost = findColorMost(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        blur = cv2.GaussianBlur(frame, (11,11), 0)
        usm = cv2.addWeighted(frame, 1.5, blur, -0.5, 0)

        gray = cv2.cvtColor(usm, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray,30,35,75)
        gray = cv2.GaussianBlur(gray, (17,17), 0)
        cv2.imshow("gray", gray)

        border = np.zeros_like(gray)
        court = np.zeros_like(frame)

        edges = cv2.Canny(gray, 100,300,apertureSize=5)
        corners = cv2.goodFeaturesToTrack(gray, 200, 0.03, 20)
        corners = np.intp(corners)
        

        filter_corners = []
        # Drawing a circle around corners
        for i in range(1, len(corners)):
            x,y = corners[i].ravel()
            onBorder = False
            for a in range(-10,11):
                for b in range(-10,11):
                    
                    try:
                        if court_mask[y+a][x+b][0] == 0 and court_mask[y+a][x+b][1] == 0 and court_mask[y+a][x+b][2] == 0:
                            onBorder = True
                            break
                    except:
                        pass

            if not onBorder:
                n=0
                for a in range(-10,11):
                    for b in range(-10,11):
                        if (hsv[y+a][x+b][1]<142).all() and (hsv[y+a][x+b][2]>200).all():
                            n+=1
                if n>50:
                    filter_corners.append([x,y])
                    cv2.circle(frame,(x,y),3,(255,50,0),-1)


        #  霍夫直線變換
        blur = cv2.GaussianBlur(edges, (3,3), 0)
        edges = cv2.addWeighted(edges, 0.5, blur, 0.5, 0)
        lines = cv2.HoughLinesP(
                edges, rho=1 , theta=np.pi / 110, threshold=150, minLineLength=50, maxLineGap=10
        )
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(court, (x1, y1), (x2, y2), (80, 20, 210), 1)

        gray2 = cv2.cvtColor(court, cv2.COLOR_BGR2GRAY)
        corners_ = cv2.cornerHarris(gray2, 5, 9, 0.04)
        corners_ = cv2.dilate(corners_, None)
        ret, corners_ = cv2.threshold(corners_, 0.01 * corners_.max(), 255, 0)
        corners_ = np.uint8(corners_)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(corners_)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        corners_ = cv2.cornerSubPix(gray2, np.float32(centroids), (5, 5), (-1, -1), criteria)

        points = []
        for i in range(1, len(corners_)):
            x,y = corners_[i]
            x = int(x)
            y = int(y)
            onBorder = False
            
            for a in range(-10,11):
                for b in range(-10,11):
                    try:
                        if court_mask[y+a][x+b][0] == 0 and court_mask[y+a][x+b][1] == 0 and court_mask[y+a][x+b][2] == 0:
                            onBorder = True
                            break
                        
                        # if not (abs(frame[y+a][x+b][0] - bgrMost[0]) < 15 and abs(frame[y+a][x+b][1] - bgrMost[1]) < 15 and abs(frame[y+a][x+b][2] - bgrMost[2]) < 15):
                        #     onBorder = True
                        #     break
                    except:
                        pass

            if not onBorder:
                cv2.circle(court,(x,y),3,(255,50,0),-1)
                points.append([x,y])




        h_lines = []
        max_length_of_line = 0
        max_y = 0
        for i in range(len(filter_corners)):
            for j in range(i+1, len(filter_corners)):
                x1, y1 = filter_corners[i]
                x2, y2 = filter_corners[j]
                #  detact the line is horizontal
                if abs(y1 - y2) < 5 and abs(x1 - x2) >5:
                    if pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5)>max_length_of_line:
                        c_x = (x1 + x2) / 2
                        for yy in range(int(min(y1, y2))-5, int(max(y1, y2))+5):
                            if edges[yy][int(c_x)] != 0:
                                break
                        else:
                            continue
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
                cv2.circle(frame, (int(p[0]), int(p[1])), 5, (77, 80, 150), 2)




            

        for p in filter_corners:
            sorted_points = sorted(points, key=lambda x: pow(pow(x[0] - p[0], 2) + pow(x[1] - p[1], 2), 0.5))
            try:
                distance = pow(pow(sorted_points[1][0] - p[0], 2) + pow(sorted_points[1][1] - p[1], 2), 0.5)
                # if distance < 50:
                cv2.circle(court,(p[0],p[1]),5,(0,255,0),-1)
            except:
                pass

        # court_ = np.zeros_like(court)
        # lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
        # min_rho=10
        # min_theta=np.pi/120
        # drew_list = []  #
        # i = 0
        # j = 0
        # draw_flag = 1
      
        # if len(lines) != 0:
        #     rho, theta = lines[0, 0] 
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a * rho
        #     y0 = b * rho
        #     x1 = int(x0 + 2000 * (-b))
        #     y1 = int(y0 + 2000 * (a))
        #     x2 = int(x0 - 2000 * (-b))
        #     y2 = int(y0 - 2000 * (a))
        #     cv2.line(frame, (x1, y1), (x2, y2), (0, 50, 255), 2)
        #     i += 1
        #     drew_list.append({'theta': theta, 'rho': rho})
    
        # for rho, theta in lines[1:, 0]: 
        #     for past_line in drew_list:
        #         theta_error = abs(past_line['theta'] - theta)
        #         rho_error = abs(past_line['rho'] - rho)
        #         if theta_error <= min_theta:
        #             if rho_error <= min_rho:
        #                 j += 1
        #                 draw_flag = 0
        #                 break 
        #     else: 
        #         drew_list.append({'theta': theta, 'rho': rho}) 
    
 
        #     if draw_flag == 1:
        #         a = np.cos(theta)
        #         b = np.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         x1 = int(x0 + 2000 * (-b))
        #         y1 = int(y0 + 2000 * (a))
        #         x2 = int(x0 - 2000 * (-b))
        #         y2 = int(y0 - 2000 * (a))
        #         cv2.line(frame, (x1, y1), (x2, y2), (0, 50, 255), 2)

        #         i += 1
        #     else:
        #         draw_flag = 1 
    
        #     pre_rho = rho 
        #     pre_theta = theta




        cv2.imshow("edges", edges)
        cv2.imshow("court", court)
        cv2.imshow("frame", frame)
        
        cv2.waitKey(0)
        
    cap.release()
cv2.destroyAllWindows()