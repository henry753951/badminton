import os
import queue
import cv2
import numpy as np
from PIL import Image, ImageDraw
import csv
import sys
import models.trajectory
from keras.models import *
import keras.backend as K
import math
BATCH_SIZE=1
HEIGHT=288
WIDTH=512
mag=1



    
def custom_loss(y_true, y_pred):
	loss = (-1)*(K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))
	return K.mean(loss)

def segment_lines(lines, deltaX, deltaY):
    h_lines = []
    v_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(y2-y1) < deltaY: # y-values are near; line is horizontal
                h_lines.append(line)
            elif abs(x2-x1) < deltaX: # x-values are near; line is vertical
                v_lines.append(line)
    return h_lines, v_lines
def getCourtLines(hsegments, vsegments): 
    h_result = [] 
    v_result = [] 
    for segment in hsegments: 
        for x1,y1,x2,y2 in segment:
            # Identified horizontal boundary lines using Brute Force
            if  (447.0 < math.dist([x1,y1] , [x2,y2]) < 449.0) or  (883.0 < math.dist([x1,y1] , [x2,y2]) < 885.0): 
                h_result.append(segment)
                
    for segment in vsegments: 
        for x1,y1,x2,y2 in segment:
            # Identified horizontal boundary lines using Brute Force
            if 450.0<math.dist([x1,y1] , [x2,y2]): 
                v_result.append(segment)   

    return h_result, v_result
q = queue.deque()
for i in range(0,8):
	q.appendleft(None)

# 1 to 800
for id in range(1, 801):
	# n = 0
	id = str(id).zfill(5)
	video_filename = F'../train/{id}/{id}.mp4'
	start_frame = 0
	currentFrame= 0
 
	# load model
	trajectory_model = load_model("trajectory_model", custom_objects={'custom_loss':custom_loss})
	# trajectory_model.summary()
	try:
		os.mkdir(F"dataset/{id}")
	except:
		pass
	cap = cv2.VideoCapture(video_filename)
 
	output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
	ret, image1 = cap.read()
	ret, image2 = cap.read()
	ret, image3 = cap.read()
	cap.set(1, currentFrame)
 
	
	# 球場
	
	court = np.zeros_like(image1)
	gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
	blur_gray = cv2.GaussianBlur(gray,(5, 5),0)
	edges = cv2.Canny(blur_gray, 50, 150, apertureSize=3)
	dilated = cv2.dilate(edges, np.ones((2,2), dtype=np.uint8))
	lines = cv2.HoughLinesP(dilated, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
	h_lines, v_lines = segment_lines(lines, 280, 0.5) 
	# filtered_h_lines, filtered_v_lines = getCourtLines(filterLines(h_lines,350, 900), filterLines(v_lines,430, 510))
 
	# for i in range(len(filtered_v_lines)): 
	# 	l = filtered_v_lines[i][0] 
	# 	cv2.line(court, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)  
		
	# # Drawing Horizontal Hough Lines on image 
	# for i in range(len(filtered_h_lines)): 
	# 	l = filtered_h_lines[i][0] 
	# 	cv2.line(court, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)  
	cv2.imshow('court', court)
	# 
	if not ret:
		continue
	ratio = image1.shape[0] / HEIGHT
	size = (int(WIDTH*ratio), int(HEIGHT*ratio))
  
	while(ret):
		current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
		if current_frame < 3:
			ret, img = cap.read()
			if not ret: 
				break
			cv2.imshow('frame',img)
			continue
	
		ret, img = cap.read()
		if not ret: 
			break
		# 球位置
		for i in range(3):
			cc_frame = (current_frame-3+i)
			x, y = 0, 0
			if i == 0:
				image = image1
			elif i == 1:
				image = image2
			elif i == 2:
				image = image3

			h_pred = models.trajectory.predict(trajectory_model,image1,image2,image3)
			if np.amax(h_pred[i]) <= 0:
				pass
			else:
				cnts, _ = cv2.findContours(h_pred[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				rects = [cv2.boundingRect(ctr) for ctr in cnts]
				max_area_idx = 0
				max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
				for i in range(len(rects)):
					area = rects[i][2] * rects[i][3]
					if area > max_area:
						max_area_idx = i
						max_area = area
				target = rects[max_area_idx]
				x, y = (int(ratio*(target[0] + target[2] / 2)), int(ratio*(target[1] + target[3] / 2)))
    
				image_cp = np.copy(image)
    

			# 路徑
			PIL_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
			PIL_image = Image.fromarray(PIL_image)
	
			if x != 0 and y != 0:
				q.appendleft([x,y])
				q.pop()
			else:
				q.appendleft(None)
				q.pop()

			for o in range(0,8):
				if q[o] is not None:
					draw_x = q[o][0]
					draw_y = q[o][1]
					bbox =  (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
					draw = ImageDraw.Draw(PIL_image)
					draw.ellipse(bbox, fill ='yellow')
					del draw
			opencvImage =  cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
			if np.amax(h_pred[i]) > 0:
				cv2.circle(opencvImage, (x, y), 5, (100,255,100), 1)
			
			cv2.imshow('frame',opencvImage)
			cv2.waitKey(10)
			
		ret, image1 = cap.read()
		ret, image2 = cap.read()
		ret, image3 = cap.read()
	print("finish")

