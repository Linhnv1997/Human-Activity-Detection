import numpy as np
import threading
import tensorflow as tf
import socket
import joblib

host = '210.123.42.177'
port = 80
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))
s.listen(2)

label = ".....Warmup...."
n_time_steps = 20



model = tf.keras.models.load_model("modellstm.h5")
loaded_rf = joblib.load("./random_forest.joblib")



def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


# def detect(model, lm_list):
#     global label
#     lm_list = np.array(lm_list)
#     lm_list = np.expand_dims(lm_list, axis=0)
#     print(lm_list.shape)
#     results = model.predict(lm_list)
#     print(results)
#     labels = ['Standing','Eating','Nothing', 'Hand Swing']
#     label = labels[np.argmax(results)]
#     return label

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    print(results)
    labels = ['Standing','Eating','Watching_TV', 'Talking_phone']
    
    label = labels[results[0]]
    return label



warmup_frames = 10

while True:
    client, addr = s.accept()
    
    
    print("Start detect....")
    try:
        print('connected by', addr)
        i = 0
        lm_list = []
        lm_list1 = []
        while True:
            i = i + 1
            data = client.recv(1024)
            str_data = data.decode("utf8")
            if str_data =="quit":
                break
            data = eval(str_data)
            data = [(x - min(data))*10 for x in data]
            lm_list.append(data)
            lm_list1 += data
            # print("Client: ",data)
            # if len(lm_list) == n_time_steps:
            #     t1 = threading.Thread(target=detect, args=(model,lm_list,))
            #     t1.start()
            #     lm_list = []
            if int(len(lm_list1)/64) == n_time_steps:
                t1 = threading.Thread(target=detect, args=(loaded_rf,lm_list1,))
                t1.start()
                lm_list1 = []
            print(label)
            #print(lm_list1)
            print("i = ",i)
            # msg = input("Server: ")
        
    finally:
        client.close()
    
    
        

s.close()

