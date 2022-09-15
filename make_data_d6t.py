import numpy as np
import threading
# import tensorflow as tf
import socket
import pandas as pd
from struct import unpack

host = '210.123.42.177'
port = 80
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))
s.listen(1)

label = "Warmup...."
n_time_steps = 10



# model = tf.keras.models.load_model("modellstm.h5")



def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    print(results)
    labels = ['Standing','Eating','Nothing', 'Hand Swing']
    label = labels[np.argmax(results)]
    return label



warmup_frames = 60
n = 0
receive = "" 
output = "" 
client, addr = s.accept()
while True:
    receive  = client.recv(20) 
    output = output+receive
    print(output)
    
    
    
        

s.close()
