# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 14:53:01 2022

@author: recep
"""

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import keras.utils as image
from keras.models import model_from_json


webcam_video_stream=cv2.VideoCapture("input-video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(webcam_video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(webcam_video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(webcam_video_stream.get(cv2.CAP_PROP_FPS))
mtcnn_detector=MTCNN()
face_exp_model = model_from_json(open("dataset/facial_expression_model_structure.json","r").read())
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')
emotions_label=('angry','disgust','fear','happy','sad','surprise','neutral')
emotions_color = [(0,0,255),(0,153,204),(0,0,0),(0,255,255),(102,102,102),(0,255,0),(255,0,0)]
emotions_count = [0,0,0,0,0,0,0]
output = cv2.VideoWriter('output-video.mp4', fourcc, fps, (width, height))



#bütün yüz konumlarını tut
all_face_locations =[]

while True:
    
    ret,current_frame = webcam_video_stream.read()
    if not ret:
        break
    #♣daha hızlı işlemek için yeniden boyutlandırma
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    all_face_locations = mtcnn_detector.detect_faces(current_frame_small)
    output_pred = np.zeros(7)
    global percentage
    percentage=0
    for index,current_face_location in enumerate(all_face_locations):
        x,y,width,height =current_face_location['box']
        left_pos=x
        top_pos=y
        right_pos=x+width
        bottom_pos=y+height
        #sığdırmak için konumu değiştir
        top_pos=top_pos*4
        right_pos=right_pos*4
        bottom_pos=bottom_pos*4
        left_pos=left_pos*4
        print('Bulunan yüz {} top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
       
        
        current_face_image=current_frame[top_pos:bottom_pos,left_pos:right_pos]
        current_face_image=cv2.cvtColor(current_face_image,cv2.COLOR_BGR2GRAY)
        current_face_image=cv2.resize(current_face_image,(48,48))
        img_pixels=image.img_to_array(current_face_image)
        img_pixels=np.expand_dims(img_pixels, axis=0)
        img_pixels/=255
       
        
        exp_predictions=face_exp_model.predict(img_pixels)
        output_pred = np.add(exp_predictions[0],output_pred)
        max_index=np.argmax(exp_predictions[0])
        emotion_label=emotions_label[max_index]
        emotion_color = emotions_color[max_index]
        emotions_count[max_index]=emotions_count[max_index]+1
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),emotion_color,2)
        font=cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame,emotion_label,(left_pos,bottom_pos),font,0.5,emotion_color,1)
        graph_width = 200
        graph_height = 100
        graph_x = 10
        graph_y = 10
    percentage = (output_pred[np.argmax(output_pred)]/len(all_face_locations)) *100
    cv2.putText(current_frame, "{}: {}".format(emotions_label[np.argmax(output_pred)],percentage),
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotions_color[np.argmax(output_pred)], 2)	    
    output.write(current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
webcam_video_stream.release()
output.release()
cv2.destroyAllWindows()




