
import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import matplotlib.pyplot as plt
import pyrealsense2 as rs

pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
profile = pipe.start(config)

label = "Warmup...."
n_time_steps = 10
lm_list = []

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("model2.h5")

frameset = pipe.wait_for_frames()
color_frame = frameset.get_color_frame()
color_init = np.asanyarray(color_frame.get_data())

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2


def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    print('=============',results)
    labels = ['Standing','Eating','Nothing', 'Hand Swing']
    label = labels[np.argmax(results)]

    # if results[0][0] > 0.5:
    #     label = "Clapping"
    # else:
    #     label = "SWING HAND"
    return label


i = 0
warmup_frames = 60
try:
    while True:
        # Store next frameset for later processing:
        frameset = pipe.wait_for_frames()
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()

        color = np.asanyarray(color_frame.get_data())
        res = color.copy()
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        l_b = np.array([24, 133, 48])
        u_b = np.array([39, 200, 181])

        mask = cv2.inRange(hsv, l_b, u_b)
        color = cv2.bitwise_and(color, color, mask=mask)

        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frameset = align.process(frameset)

        # Update color and depth frames:
        aligned_depth_frame = frameset.get_depth_frame()
        colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

        ### motion detector
        d = cv2.absdiff(color_init, color)
        gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=3)
        (c, _) = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(color, c, -1, (0, 255, 0), 2)
        color_init = color

        depth = np.asanyarray(aligned_depth_frame.get_data())

        for contour in c:
            if cv2.contourArea(contour) < 1500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            bottomLeftCornerOfText = (x, y)

            # Crop depth data:
            depth = depth[x:x + w, y:y + h].astype(float)

            depth_crop = depth.copy()

            if depth_crop.size == 0:
                continue
            depth_res = depth_crop[depth_crop != 0]

            # Get data scale from the device and convert to meters
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth_res = depth_res * depth_scale

            if depth_res.size == 0:
                continue

            dist = min(depth_res)

            cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 3)
            text = "Depth: " + str("{0:.2f}").format(dist)
            cv2.putText(res,
                        text,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

        # cv2.namedWindow('RBG', cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('Depth', colorized_depth)
        # cv2.namedWindow('mask', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('mask', mask)

        imgRGB = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        # cv2.imshow('RBG', res)
        results = pose.process(imgRGB)
        i = i + 1
        if i > warmup_frames:
            print("Start detect....")

            if results.pose_landmarks:
                c_lm = make_landmark_timestep(results)

                lm_list.append(c_lm)
                if len(lm_list) == n_time_steps:
                    # predict
                    t1 = threading.Thread(target=detect, args=(model, lm_list,))
                    t1.start()
                    lm_list = []

                res = draw_landmark_on_image(mpDraw, results, res)
                # colorized_depth = draw_landmark_on_image(mpDraw, results, colorized_depth)

        res = draw_class_on_image(label, res)
        colorized_depth = draw_class_on_image(label, colorized_depth)
        # cv2.imshow("Image", res)
        cv2.imshow('Depth', colorized_depth)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipe.stop()
