import numpy as np
import cv2
from math import atan, pi
import math
def is_in_court(line, court):
    '''
    判斷球是否在球場內
    @line: (x1,y1,x2,y2)
    @court: np.array of court
    @return: bool

    '''
    x1,y1,x2,y2 = line
    court = cv2.cvtColor(court, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(court, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    if(len(contours) == 0):
        return True
    dist = cv2.pointPolygonTest(contours[0], (int(x1), int(y1)), True)
    if dist < 20:
        return False
    dist = cv2.pointPolygonTest(contours[0], (int(x2), int(y2)), True)
    if dist < 20:
        return False
    return True
    
def interpolate(pt):
    '''
    將兩點之間的座標點都算出來
    @pt: (x1,y1,x2,y2)
    @return: [(x,y)...]
    '''
    x1,y1,x2,y2 = pt
    num = max(abs(x1-x2), abs(y1-y2))
    np.around(np.linspace(x1, x2+1, num=num+1))
    return np.stack((np.around(np.linspace(x1, x2, num=num+1)), np.around(np.linspace(y1, y2, num=num+1))), axis=1).astype(int)
def check_white(hsv, lines, set_i):
    line_mask = np.zeros(hsv.shape[:2], 'uint8')
    for line_id in set_i:
        x1,y1,x2,y2 = lines[line_id]
        cv2.line(line_mask,(x1,y1),(x2,y2),(255,255,255),1)
    line_mask = cv2.dilate(line_mask.astype(np.float32), None, iterations=2) > 0
#     cv2.imshow('line_mask', line_mask.astype('uint8')*255)
#     cv2.waitKey(0)
    
    white_hsv = np.uint8([[[0,0,255]]])
    range1 = [20,255,200]
    range2 = [20,0,0]
    white_region = cv2.inRange(hsv, white_hsv-[0,0,60], white_hsv+[255,60,0]) > 0
#     cv2.imshow('white_region', white_region.astype('uint8')*255)
#     cv2.waitKey(0)
    white_line_region = np.logical_and(line_mask, white_region)
#     cv2.imshow('white_line_region', white_line_region.astype('uint8')*255)
#     cv2.waitKey(0)
    
    cv2.destroyAllWindows()
#     print((white_line_region.sum() / line_mask.sum()))
    return (white_line_region.sum() / line_mask.sum()) > 0.03

def find_edge(p1, p2, h, w):
    '''
    找出兩點在邊界上的點
    @p1: (x1,y1)
    @p2: (x2,y2)
    @h: height
    @w: width
    @return: (edge_p1, edge_p2)
    '''
    v12 = p2-p1 + 1e-18
    target = np.array([0 if v12[0] < 0 else w-1, 0 if v12[1] < 0 else h-1])
    steps = (target - p2)/v12
    edge_p2 = (steps.min()*v12+p2).astype(int)
    
    v21 = -v12
    target = np.array([0 if v21[0] < 0 else w-1, 0 if v21[1] < 0 else h-1])
    steps = (target - p1)/v21
    edge_p1 = (steps.min()*v21+p1).astype(int)
    if edge_p1[1] == 0: # start with top edge
        return edge_p1.astype(int), edge_p2.astype(int)
    elif edge_p1[0] == 0: # start with left edge
        if edge_p2[1] == 0: # end with top edge
            return edge_p2.astype(int), edge_p1.astype(int)
        else:
            return edge_p1.astype(int), edge_p2.astype(int)
    elif edge_p1[1] == h-1: # start with botton edge
        if edge_p2[0] == w-1: # end with right edge
            return edge_p1.astype(int), edge_p2.astype(int)
        else:
            return edge_p2.astype(int), edge_p1.astype(int)
    elif edge_p1[0] == w-1: # start with right edge
        return edge_p2.astype(int), edge_p1.astype(int)

def adjust_gamma(image, gamma=1.0):
    '''
    調整圖片的gamma值
    @image: 圖片
    @gamma: gamma值
    @return: 調整後的圖片
    '''
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table) # apply gamma correction using the lookup table

def find_intersection(l1, l2):
    '''
    找出兩條線的交點
    @l1: (x1,y1,x2,y2)
    @l2: (x1,y1,x2,y2)
    @return: (x,y)
    '''
    if len(l1) == 2:
        l1 = np.array([l1[0][0],l1[0][1],l1[1][0],l1[1][1]])
    if len(l2) == 2:
        l2 = np.array([l2[0][0],l2[0][1],l2[1][0],l2[1][1]])
    with np.errstate(divide='ignore', invalid='ignore'):
        # s * (x - px) + py = y
        s1 = (l1[3] - l1[1]) / (l1[2] - l1[0])
        s2 = (l2[3] - l2[1]) / (l2[2] - l2[0])
        s1 = 1e+20 if np.isinf(s1) else s1
        s2 = 1e+20 if np.isinf(s2) else s2
        px1 = l1[0]
        py1 = l1[1]
        px2 = l2[0]
        py2 = l2[1]
        # [s1 * (x - px1) + py1] - [s2 * (x - px2) + py2] = 0
        x = (-py1+py2 + s1*px1 - s2*px2) / (s1-s2)
        x = 1e+6 if np.isnan(x) else x
        y = s1 * (x - px1) + py1 if (x - px1) != 0 else s2 * (x - px2) + py2
        return (x,y)

def merge_lines(lines,threshold=25):
    '''
    合併重複的線段
    @lines: [(x1,y1,x2,y2), ...]
    @return: [(x1,y1,x2,y2), ...]
    '''
    
    num_l = len(lines)
#     remove = []
    for l1 in range(num_l):
        for l2 in range(l1+1, num_l):
            if (lines[l1] == lines[l2]).all() or (lines[l1] == np.roll(lines[l2], 2)).all():
                lines[l2] = np.array([0,0,0,0])
            else:
                dist = np.linalg.norm(lines[l1][:2] - lines[l2][:2]) + np.linalg.norm(lines[l1][2:] - lines[l2][2:])
                if dist <= threshold:
                    lines[l2] = np.array([0,0,0,0])
    lines = [i for i in lines if not (i==np.array([0,0,0,0])).all()]
    return lines

def dist(samples):
    '''
    計算兩兩點之間的距離
    @samples: (x1, y1, x2, y2)
    @return: distance
    '''
    xx,xy = np.meshgrid(samples[:,0], samples[:,0])
    yx,yy = np.meshgrid(samples[:,1], samples[:,1])
    ans = np.sqrt((xy-xx)**2+(yx-yy)**2)
    xx,xy = np.meshgrid(samples[:,2], samples[:,2])
    yx,yy = np.meshgrid(samples[:,3], samples[:,3])
    ans = np.maximum(ans, np.sqrt((xy-xx)**2+(yx-yy)**2))
    
    ans = np.triu(ans)
    ans[ans==0] = ans.max()+1
    return ans


def parse_color(strcolor):
    strcolor = strcolor[4:-1].split(',')
    strcolor.reverse()
    return tuple(map(int,strcolor))


def getSlope(line):
    '''
    計算線段的斜率
    @line: (x1,y1,x2,y2)
    @return: slope
    '''
    if line[2]-line[0] == 0:
        return 1e+20
    else:
        return (line[3]-line[1])/(line[2]-line[0])

def distance_to_line(line, x0, y0):
    """
    計算點 (x0,y0) 到直線的距離
    @line: 直線的兩個端點坐標，格式為 [x1, y1, x2, y2]
    @x0, y0: 點的坐標
    @return: 點到直線的距離
    """
    # 將直線的兩個端點坐標分別賦值給變量 x1, y1, x2, y2
    x1, y1, x2, y2 = line
    
    # 計算直線的係數 a, b, c
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    
    # 計算點到直線的距離
    return abs(a * x0 + b * y0 + c) / np.sqrt(a ** 2 + b ** 2)


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
    def get_intersections(line1, line2):
        A = np.array(line1)
        B = np.array(line2)
        t, s = np.linalg.solve(np.array([A[1]-A[0], B[0]-B[1]]).T, B[0]-A[0])    
        return (1-t)*A[0] + t*A[1]
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



