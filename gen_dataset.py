import cv2
import pandas as pd
import random,os
import numpy as np



# =========== config =================
prue_csv_mode = True

# 最大移動偵測範圍及嘗試追蹤次數
max_distacne = 500000
MaxTryToTrack = 10

dataset_dir = "ball_dataset"
# ====================================


try:
    os.mkdir(dataset_dir)
except:
    pass

def remove_Contour_with_area(mask,n : int,b : int = -1):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > n and (area < b or b == -1):
            cv2.drawContours(mask, [cnt],0, 255, -1)
    return mask

# 1 to 800
for id in range(1, 801):
    # n = 0
    nn = 0
    id = str(id).zfill(5)
    video_filename = F'../train/{id}/{id}.mp4'
    start_frame = 0
    try:
        os.mkdir(F"{dataset_dir}/{id}")
    except:
        pass
    cap = cv2.VideoCapture(video_filename)
    
    
    # csv draw setting
    data = pd.read_csv(F'../train/{id}/{id}_S2.csv')
    df = pd.DataFrame(columns=["Frame", "x", "y"])
    color_yellow = (0, 255, 255)
    color_green = (0, 255, 0)
    radius = 5
    #
    
    
    # 建立背景差分器
    backSub = cv2.createBackgroundSubtractorKNN( history=200,  dist2Threshold=800,detectShadows=False)
    
    
    
    
    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_final = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        mask = backSub.apply(frame)     
        
        kernel = np.ones((4,4),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        mask = cv2.dilate(mask, None)     
        
        mask = cv2.GaussianBlur(mask, (15, 15),0)     
        
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, None)
        thresh = cv2.GaussianBlur(thresh, (21, 21),0)
       
       
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        body_mmask = np.zeros_like(mask)
        if len(contours)!=0:
            mmask = np.zeros_like(mask)
            for cnt in contours:
                if cv2.contourArea(cnt) <=2500:
                    cv2.drawContours(mmask, [cnt], 0, 255, -5)
                else:
                    cv2.drawContours(body_mmask, [cnt], 0, 255, -1)
                    
        

        # 

        
        contours2, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours2:
            cv2.drawContours(body_mmask, [cnt], 0, 255, -5)
        body_mmask = cv2.dilate(body_mmask, None)     

        body_mmask = remove_Contour_with_area(body_mmask, 1000)
        body_mmask = cv2.GaussianBlur(body_mmask, (79, 79),0)
        body_mmask = remove_Contour_with_area(body_mmask, 1000)
        
        # cv2.imshow('body_mmask', body_mmask)
        # 
        # cv2.imshow('mmask', mmask)
        # 
        
        cv2.subtract(mask, body_mmask, mask)
        cv2.bitwise_and(mmask, mask, mmask)
        
        
        mmask_ = remove_Contour_with_area(mmask, 30)
        cv2.imshow('mmask', mmask_)
        contours, _ = cv2.findContours(mmask_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final = np.zeros_like(mask)
        framecp = frame.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            if(x>300 and x< 1000):
                cv2.drawContours(final, [cnt],0, 255, -1)
            
        
        cv2.imshow('final', final)
        ball_contours, _ = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 上次座標
        last_coordinate = [0,0]
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame >= data.loc[0, 'HitFrame']+5 and current_frame <= data.loc[len(data)-1, 'HitFrame']:
            if len(ball_contours) != 0:
                x_,y_,w_,h_ = cv2.boundingRect(ball_contours[0])
                x_ = x_ + int(w_/2)
                y_ = y_ + int(h_/2)
                distance =  pow(x_-last_coordinate[0],2)+pow(y_-last_coordinate[1],2)
                for cnt in ball_contours:
                    x,y,w,h = cv2.boundingRect(cnt)
                    _x = x + int(w/2)
                    _y = y + int(h/2)
                    curr_distance = pow(_x-last_coordinate[0],2)+pow(_y-last_coordinate[1],2)
                    if curr_distance <= distance:
                        if curr_distance<=max_distacne or nn>=MaxTryToTrack:
                            nn=0
                            distance = curr_distance
                            roi = frame[y:y+h, x:x+w]
                            cv2.circle(framecp, (int(x+w/2),int(y+h/2)), 5, (0,0,255), -1)            
                            cv2.imshow('roi', cv2.resize(roi, (w*5, h*5)))
                            if(prue_csv_mode==False):
                                cv2.imwrite(F'{dataset_dir}/{id}/{current_frame}_{_x}_{_y}.jpg', roi)
                            df.loc[len(df)] = (current_frame,_x,_y)
                        
                            break
                        else:
                            nn+=1
            


        prev_frame = frame
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        
        cv2.imshow('Frame', framecp)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    df.to_csv(F"{dataset_dir}/{id}/dataset.csv", index=False)
    cap.release()
    cv2.destroyAllWindows()