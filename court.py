import cv2
import numpy as np
import os,math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pprint
import random
import utils 
pp = pprint.PrettyPrinter(indent=4)
# 讀取範例影片
video = cv2.VideoCapture('match.mp4')

# 記錄目前處理到第幾幀(從0開始)
currentFrame = 0





for id in range(224, 801):
    id = str(id).zfill(5)
    video_filename = F'../train/{id}/{id}.mp4'
    video = cv2.VideoCapture(video_filename)

    success,frame = video.read()
    if not success:
        print(F"Cannot read video file {id}")
        break

    try:
        cv2.destroyWindow("court_mask")
    except:
        pass
        corner = utils.getPerspectiveTransformMatrix(frame,  show_frame = False)
        easy_court = True
        if len(corner) == 4:
            court_mask = np.zeros_like(frame)
            cv2.fillPoly(court_mask, np.array([corner], dtype=np.int32), (255, 255, 255))
            # cv2.imshow("court_mask", court_mask)
            frame = cv2.bitwise_and(frame, court_mask)
            Canny_low_threshold = 85
            Canny_high_threshold = 85*3
            rho = 1  # distance resolution in pixels of the Hough grid
            theta = np.pi / 180  # angular resolution in radians of the Hough grid
            threshold = 30  # minimum number of votes (intersections in Hough grid cell)
            min_line_length = 100  # minimum number of pixels making up a line
            max_line_gap = 5  # maximum gap in pixels between connectable line segments
        else:
            easy_court = False
            pts = np.array([[[0, 0], [0, 0], [0, 0], [0, 0]]])
            court_mask = np.ones_like(frame)
            contrast =50
            brightness = 40
            frame = frame * (contrast/127 + 1) - contrast + brightness
            frame = np.clip(frame, 0, 255)
            frame = np.uint8(frame)
            # 角點 與 霍夫線段參數
            Canny_low_threshold = 15
            Canny_high_threshold = 85*3
            rho = 1  # distance resolution in pixels of the Hough grid
            theta = np.pi / 180  # angular resolution in radians of the Hough grid
            threshold = 150  # minimum number of votes (intersections in Hough grid cell)
            min_line_length = 60  # minimum number of pixels making up a line
            max_line_gap = 20  # maximum gap in pixels between connectable line segments    
    
    image = frame.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray,30,35,75)
    kernel_size = 5

    # sharpen
    gray = utils.adjust_gamma(gray, 0.4)
    blur = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0).astype(int)
    sub = gray.astype(int) - blur
    sharped_img = np.clip(gray.astype(int) + sub*5, a_min = 0, a_max = 255).astype('uint8')
    cv2.imshow('sharped_img', sharped_img)
    # sharped_gray = cv2.cvtColor(sharped_img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(sharped_img, Canny_low_threshold, Canny_high_threshold, apertureSize=3)

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    mask = court_mask > 0
    mask = np.stack((mask,mask,mask), 2)
    m_lines = []


    # test
    test = image.copy()
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        cv2.line(test, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imshow('test', test)
    cv2.imshow('edge', edges)


    # filter out lines
    for line in lines.reshape(-1,4):
        # check if line is in court
        if not utils.is_in_court(line, court_mask):
            continue
        full_line = utils.interpolate(line)
        check = mask[:, :,0][full_line[:,1], full_line[:,0]]
        if check.sum() >= len(check) * 0.5:
            m_lines.append(line)
            
    print(len(m_lines), 'Lines')



    _h, _w, _ = image.shape
    e_m_lines = []
    for x1,y1,x2,y2 in m_lines:
        (e_x1,e_y1), (e_x2,e_y2) = utils.find_edge(np.array([x1,y1]), np.array([x2,y2]), _h, _w)
        e_m_lines.append(np.array([e_x1,e_y1,e_x2,e_y2]))
    e_m_lines.sort(key=lambda ele: ele[1])
    e_m_lines = utils.merge_lines(e_m_lines,25 if easy_court else 80)




    horizontal_lines = [[x1,y1,x2,y2] for x1,y1,x2,y2 in e_m_lines if (abs(y1-y2)<=50).all()]
    vertical_lines = [[x1,y1,x2,y2] for x1,y1,x2,y2 in e_m_lines if (abs(y1-y2)>50).all()]
    h_slopes = [(line[1]-line[3]) / (line[0]-line[2]) for line in horizontal_lines]

    
    
    if easy_court:  #若太多雜縣 mean 不準
        mean = np.mean(h_slopes)
        horizontal_lines = sorted([horizontal_lines[index] for index,s in enumerate(h_slopes) if abs(s-mean) <= 0.02],key=lambda ele: ele[1], reverse=True)
        # 找出有沒有相交的線 把離mean slope遠的線刪掉
        for line in horizontal_lines:
            for line2 in horizontal_lines:
                if line[1] < line2[1] and line[3] > line2[3]:
                    if not abs((line[1]-line[3]) / (line[0]-line[2]) - mean) > abs((line2[1]-line2[3]) / (line2[0]-line2[2]) - mean):
                        horizontal_lines.remove(line2)
    else: 
        horizontal_lines = sorted(horizontal_lines,key=lambda ele: ele[1], reverse=True)
                

    distance_each_line = []
    for i, line in enumerate(horizontal_lines):
        x1, y1, x2, y2 = line
        if i < len(horizontal_lines) - 1:
            next_line = horizontal_lines[i + 1]
            next_x, next_y = (next_line[0] + next_line[2]) / 2, (next_line[1] + next_line[3]) / 2
            distance = utils.distance_to_line(line, next_x, next_y)
            distance_each_line.append(distance)

    # 找第一個前後差xxScale倍以上的
    # Todo : 投影轉換 在量比例
    xxScale = 3 if easy_court else 2
    f=0
    print(distance_each_line)
    for i, distance in enumerate(distance_each_line):
        if i < len(distance_each_line) - 1:
            print(distance_each_line[i + 1]/distance )
            if distance_each_line[i + 1]/distance > xxScale:
                f = i
                break

    horizontal_lines = horizontal_lines[f:f+3]

    temp = []
    center_line = None
    if len(vertical_lines) != 4 or len(vertical_lines) != 5:
        height_, width_, _ = image.shape
        center = int(width_/2)
        cy = int(height_/2)
        
        for i in range(int(1080/2) + 100):
            rx = center+i
            lx = center-i
            cv2.circle(image, (rx,cy), 5, (0,255,0), -1)
            cv2.circle(image, (lx,cy), 5, (0,255,0), -1)
            cv2.imshow("123",image)
            cv2.waitKey(1)
            for line in vertical_lines:
                if(abs(line[0]-line[2]) < 60):
                    center_line = line
                if (utils.distance_to_line(line,rx,cy) <=1 or utils.distance_to_line(line,lx,cy)<=1) and abs(line[0]-line[2]) > 100:
                    if temp.count(line) == 0:
                        temp.append(line)
                if len(temp) == 4:
                    break
            else:
                continue
            break
        vertical_lines = temp




    temp1 = []
    temp2 = []
    court_points = {
        "corner":[],
        "center":[],
    }
    for line in horizontal_lines:
        x1,y1,x2,y2 = line
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),3)
        for line2 in vertical_lines:
            x,y = utils.find_intersection(line, line2)
            # check x,y in range 
            height_, width_, _ = image.shape
            if x >= 0 and x < width_ and y >= 0 and y < height_:
                court_points["corner"].append([x,y])
                if temp1.count(line) == 0:
                    temp1.append(line)
                if temp2.count(line2) == 0:
                    temp2.append(line2)
                cv2.circle(image, (int(x),int(y)), 5, (0,255,0), 5)
    horizontal_lines = temp1
    vertical_lines = temp2


    if center_line != None:
        l = 0
        r = 0
        for p in court_points["corner"]:
            x,y = p
            if center_line[0] > x:
                l += 1
            elif center_line[0] < x:
                r += 1
        if l==6 and r==6:
            for line in horizontal_lines:
                x,y = utils.find_intersection(center_line, line)
                court_points["center"].append([x,y])
                cv2.line(image,(center_line[0],center_line[1]),(center_line[2],center_line[3]),(250,150,250),2)


    if len(court_points["corner"])==12:
        cv2.putText(image, ": )", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 255), 2, cv2.LINE_AA)
    # else:
    #     cv2.putText(image, ": (", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 255), 2, cv2.LINE_AA)



    for x1,y1,x2,y2 in horizontal_lines:
        color = random.choice([(255,0,255)])
        cv2.line(image,(x1,y1),(x2,y2),color,1)
    for x1,y1,x2,y2 in vertical_lines:
        color = random.choice([(255,0,0)])
        cv2.line(image,(x1,y1),(x2,y2),color,1)

    cv2.putText(image, id, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 
    cv2.imshow("123",image)
    cv2.waitKey(0)
    

    # 釋放影片資源
    video.release()