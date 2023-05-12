import os
import queue
import cv2
import numpy as np
from PIL import Image, ImageDraw
import csv
import sys
import models.trajectory
# from keras.models import *
# import keras.backend as K
import math
# import tensorflow as tf
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.keras.utils.disable_interactive_logging()
BATCH_SIZE = 1
HEIGHT = 288
WIDTH = 512
mag = 1


# def custom_loss(y_true, y_pred):
#     loss = (-1) * (
#         K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1))
#         + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1))
#     )
#     return K.mean(loss)


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


def getCourtLines(hsegments, vsegments):
    h_result = []
    v_result = []
    for segment in hsegments:
        for x1, y1, x2, y2 in segment:
            # Identified horizontal boundary lines using Brute Force
            if (447.0 < math.dist([x1, y1], [x2, y2]) < 449.0) or (
                883.0 < math.dist([x1, y1], [x2, y2]) < 885.0
            ):
                h_result.append(segment)

    for segment in vsegments:
        for x1, y1, x2, y2 in segment:
            # Identified horizontal boundary lines using Brute Force
            if 450.0 < math.dist([x1, y1], [x2, y2]):
                v_result.append(segment)

    return h_result, v_result


def getSpeed(pos1: list, pos2: list,frame_num : int):
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt(pow((y2 - y1),2)+pow((x2 - x1),2))*30/frame_num,(y2 - y1)*30/frame_num,(x2 - x1)*30/frame_num

def getVector(pos1: list, pos2: list):
    vec_AB = (pos2[0] - pos1[0], pos2[1] - pos1[1])
    vec_AB_len = math.sqrt(vec_AB[0]**2 + vec_AB[1]**2)
    if vec_AB_len == 0:
        unit_vec_AB = (0, 0)
    else:
        unit_vec_AB = (vec_AB[0] / vec_AB_len, vec_AB[1] / vec_AB_len)
    return unit_vec_AB



checkpoint = torch.load("model_best.pt")
param_dict = checkpoint['param_dict']
model_name = param_dict['model_name']
num_frame = param_dict['num_frame']
input_type = param_dict['input_type']
model = models.trajectory.get_model(model_name, num_frame, input_type).cuda()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 1 to 800
for id in range(1, 801):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(F'new-{id}.mp4', fourcc, 30.0, (1280 ,720))
    # n = 0
    id = str(id).zfill(5)
    video_filename = f"../train/{id}/{id}.mp4"
    start_frame = 0
    currentFrame = 0

    q = queue.deque()
    for i in range(0, 8):
        q.appendleft(None)
    pos_3 = queue.deque()



    try:
        os.mkdir(f"dataset/{id}")
    except:
        pass
    cap = cv2.VideoCapture(video_filename)

    output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, image1 = cap.read()
    ret, image2 = cap.read()
    ret, image3 = cap.read()
    cap.set(1, currentFrame)
    last_coordinate = [0,0]
    # 球場

    court = np.zeros_like(image1)
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur_gray, 50, 150, apertureSize=3)
    dilated = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8))
    lines = cv2.HoughLinesP(
        dilated, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10
    )
    h_lines, v_lines = segment_lines(lines, 280, 0.5)
    # filtered_h_lines, filtered_v_lines = getCourtLines(filterLines(h_lines,350, 900), filterLines(v_lines,430, 510))

    for i in range(len(v_lines)):
        l = v_lines[i][0]
        cv2.line(court, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    # Drawing Horizontal Hough Lines on image
    for i in range(len(h_lines)):
        l = h_lines[i][0]
        cv2.line(court, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow("court", court)
    #
    if not ret:
        continue
    ratio = image1.shape[0] / HEIGHT
    size = (int(WIDTH * ratio), int(HEIGHT * ratio))

    while ret:
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame < 3:
            ret, img = cap.read()
            if not ret:
                break
            cv2.imshow("frame", img)
            out.write(img)
            continue

        ret, img = cap.read()

    
        if not ret:
            break
        # 球位置

        frame_num_t=1
        color = (100, 255, 200)
        www = 2
        vec = [0, 0]
        speed = 0
        nn = 0
        for i in range(3):
            cc_frame = current_frame - 3 + i
            x, y = 0, 0
            if i == 0:
                image = image1
            elif i == 1:
                image = image2
            elif i == 2:
                image = image3

            image_cp = np.copy(image)
            h_pred = models.trajectory.predict(model, [image1, image2, image3],3)
            
            if np.amax(h_pred[i]) <= 0:
                pass
            else:
                cnts, _ = cv2.findContours(
                    h_pred[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                distance = int(math.sqrt(pow(rects[0][0]-last_coordinate[0],2)+pow(rects[0][1]-last_coordinate[1],2)))
                curr_distance = None
                max_area_idx = -1
                max_area = rects[0][2] * rects[0][3]
                for i in range(len(rects)):
                    curr_distance = int(math.sqrt(pow(rects[i][0]-last_coordinate[0],2)+pow(rects[i][1]-last_coordinate[1],2)))
                    area = rects[i][2] * rects[i][3]
                    if area >= max_area:
                        # 1.5 考慮球揮拍後速度差太多
                        # 之後修改成在玩家附近就不要考慮速度的推算
                        # 還有會吃到其他場地的球
                        # 之後修改成距離算法
                        if curr_distance <= speed/30 * 1.5 or speed/30 == 0.0 or nn>1:
                            nn = 0
                            last_coordinate = [rects[i][0], rects[i][1]]
                            curr_distance = distance
                            max_area_idx = i
                if max_area_idx == -1:
                    x, y = (
                        0,
                        0,
                    )
                    nn+=1
                else:
                    target = rects[max_area_idx]
                    # print(target)
                    x, y = (
                        int(ratio * (target[0] + target[2] / 2)),
                        int(ratio * (target[1] + target[3] / 2)),
                    )

            # 路徑
            
            if x != 0 and y != 0:
                q.appendleft([x, y])
                pos_3.appendleft([x, y])
                q.pop()
                frame_num_t = 1
            else:
                frame_num_t+=1
                
                q.appendleft(None)
                q.pop()

 
            


            if len(pos_3) >= 3:
                
                speed = getSpeed([pos_3[0][0], pos_3[0][1]], [pos_3[1][0], pos_3[1][1]],frame_num_t)[0]
                vec = getVector([pos_3[0][0], pos_3[0][1]], [pos_3[1][0], pos_3[1][1]])
                cv2.putText(
                    image_cp,
                    "vec: " + str(vec),
                    (10, 50),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image_cp,
                    "frame_num: " + str(frame_num_t),
                    (10, 100),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image_cp,
                    "Speed: " + str(speed/30),
                    (10, 250),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                if len(pos_3) > 3:
                    pos_3.pop()



            PIL_image = cv2.cvtColor(image_cp, cv2.COLOR_BGR2RGB)
            PIL_image = Image.fromarray(PIL_image)
            for o in range(0, 8):
                if q[o] is not None:
                    draw_x = q[o][0]
                    draw_y = q[o][1]
                    bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
                    draw = ImageDraw.Draw(PIL_image)
                    draw.ellipse(bbox, fill="yellow")
                    del draw
            opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
            if x != 0 and y != 0:
                if i < len(h_pred) and np.amax(h_pred[i]) > 0:
                    power = abs(speed/30)
                    cv2.circle(opencvImage, (x, y), 5, color, www)
                    cv2.circle(opencvImage, (x, y), int(power), (255,255,100), 1)
                    cv2.line(opencvImage, (x, y), (int(x + vec[0]*-power),int( y + vec[1]*-power)), (100, 50, 230), 3)

            cv2.imshow("frame", opencvImage)
            out.write(opencvImage)
            cv2.waitKey(1)
        ret, image1 = cap.read()
        ret, image2 = cap.read()
        ret, image3 = cap.read()
    print("finish")
    cap.release()
    # out.release()
