import cv2
import pandas as pd
import random
import numpy as np

# 1 to 800
for id in range(1, 801):
    id = str(id).zfill(5)
    n = 0
    video_filename = F'train/{id}/{id}.mp4'
    start_frame = 0
    data = pd.read_csv(F'train/{id}/{id}_S2.csv')


    cap = cv2.VideoCapture(video_filename)

    color_yellow = (0, 255, 255)
    color_green = (0, 255, 0)
    radius = 5
    draw=False
    row = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if row < len(data) and current_frame >= data.loc[row, 'HitFrame'] and current_frame <= data.loc[row, 'HitFrame']+15:
            hitter_x = int(data.loc[row, 'HitterLocationX'])
            hitter_y = int(data.loc[row, 'HitterLocationY'])
            DefenderLocationX = int(data.loc[row, 'DefenderLocationX'])
            DefenderLocationY = int(data.loc[row, 'DefenderLocationY'])
            if(row>=1):
                landing_x = int(data.loc[row-1, 'LandingX'])
                landing_y = int(data.loc[row-1, 'LandingY'])
                cv2.circle(frame, (landing_x, landing_y), radius, color_yellow, -1)
            cv2.circle(frame, (hitter_x, hitter_y), radius, color_green, -1)
            cv2.circle(frame, (DefenderLocationX, DefenderLocationY), radius, color_green, -1)
            # print ShotSeq,Hitter,RoundHead,Backhand,BallHeight,BallType,Winner by data.loc[row,{key}]
            if current_frame == data.loc[row, 'HitFrame']+1 :
                print(data.loc[row, 'ShotSeq'],data.loc[row, 'Hitter'],data.loc[row, 'RoundHead'],data.loc[row, 'Backhand'],data.loc[row, 'BallHeight'],data.loc[row, 'BallType'],data.loc[row, 'Winner'])
            draw=True
        elif(draw):
            draw = False
            row+=1


        cv2.imshow('Frame2', frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()