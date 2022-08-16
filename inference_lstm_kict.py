import numpy as np
import threading
import tensorflow as tf
import socket

host = '210.123.42.177'
port = 80
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))
s.listen(2)

label = "Warmup...."
n_time_steps = 10



model = tf.keras.models.load_model("modellstm.h5")



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

while True:
    client, addr = s.accept()
    
    
    print("Start detect....")
    try:
        print('connected by', addr)
        i = 0
        lm_list = []
        while True:
            i = i + 1
            data = client.recv(1024)
            str_data = data.decode("utf8")
            if str_data =="quit":
                break
            data = eval(str_data)
            data = [x - min(data) for x in data]
            lm_list.append(data)
            # print("Client: ",data)
            if len(lm_list) == n_time_steps:
                t1 = threading.Thread(target=detect, args=(model,lm_list,))
                t1.start()
                lm_list = []
            print(label)
            print("i = ",i)
            # msg = input("Server: ")
        
    finally:
        client.close()
    
    
        

s.close()

