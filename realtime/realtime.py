# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 14:53:01 2022

@author: recep
"""

import cv2
import numpy as np
from keras.api.preprocessing import image
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras


#start video recording
webcam_video_stream=cv2.VideoCapture(0)
#create mtcnn instance
mtcnn_detector=MTCNN()
#load model
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(
        filters=64, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='uniform', mode='fan_avg', scale=1.0),
        bias_initializer='zeros', input_shape=(48, 48, 1), name='conv2d_1'
    ),
    layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2), padding='valid', name='max_pooling2d_1'),
    layers.Conv2D(
        filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='uniform', mode='fan_avg', scale=1.0),
        bias_initializer='zeros', name='conv2d_2'
    ),
    layers.Conv2D(
        filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='uniform', mode='fan_avg', scale=1.0),
        bias_initializer='zeros', name='conv2d_3'
    ),
    layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='average_pooling2d_1'),
    layers.Conv2D(
        filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='uniform', mode='fan_avg', scale=1.0),
        bias_initializer='zeros', name='conv2d_4'
    ),
    layers.Conv2D(
        filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='uniform', mode='fan_avg', scale=1.0),
        bias_initializer='zeros', name='conv2d_5'
    ),
    layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='average_pooling2d_2'),
    layers.Flatten(name='flatten_1'),
    layers.Dense(
        units=1024, activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='uniform', mode='fan_avg', scale=1.0),
        bias_initializer='zeros', name='dense_1'
    ),
    layers.Dropout(rate=0.2, name='dropout_1'),
    layers.Dense(
        units=1024, activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='uniform', mode='fan_avg', scale=1.0),
        bias_initializer='zeros', name='dense_2'
    ),
    layers.Dropout(rate=0.2, name='dropout_2'),
    layers.Dense(
        units=7, activation='softmax',
        kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='uniform', mode='fan_avg', scale=1.0),
        bias_initializer='zeros', name='dense_3'
    )
])

# Modeli JSON formatına çevirme
config = model.to_json()

# Modeli JSON formatından yükleme
loaded_model = tf.keras.models.model_from_json(config)


# Model özetini gösterelim
model.summary()

model.load_weights('../model/model_ready/facial_expression_model_weights.h5')
#emotions
emotions_label=('angry','disgust','fear','happy','sad','surprise','neutral')
#emotions_color
emotions_color = [(0,0,255),(0,153,204),(0,0,0),(0,255,255),(102,102,102),(0,255,0),(255,0,0)]
#emotions_count
emotions_count = [0,0,0,0,0,0,0]




#bütün yüz konumlarını tut
all_face_locations =[]

while True:
    
    ret,current_frame = webcam_video_stream.read()
    #♣resize frame for faster face recognition
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    all_face_locations = mtcnn_detector.detect_faces(current_frame_small)
    #create empty array for storing emotions of face in frame
    output_pred = np.zeros(7)
    global percentage
    percentage=0
    for index,current_face_location in enumerate(all_face_locations):
        x,y,width,height =current_face_location['box']
        left_pos=x
        top_pos=y
        right_pos=x+width
        bottom_pos=y+height
        #change location coordinates to original size
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
       
        
        exp_predictions=model.predict(img_pixels)
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
    cv2.imshow("Yuz Tanima",current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
webcam_video_stream.release()
cv2.destroyAllWindows()




