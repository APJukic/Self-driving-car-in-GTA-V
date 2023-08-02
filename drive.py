"""
Description: This script enables autonomous driving for the car using a pre-trained steering prediction model known as PilotNet. The model accurately predicts the car's steering angle by analyzing the visual input captured by the driver. Based on these predictions, the script navigates the car to turn left, right, or move straight ahead, ensuring that the car is able to stay in its lane while driving.
"""

import tensorflow as tf
import numpy as np
from pyautogui import screenshot
import cv2
import time
import os
import sys
sys.path.append('./keyboard')
from directkeys import PressKey, ReleaseKey, W, A, S, D

def forward():
    ReleaseKey(A)
    ReleaseKey(D)
    PressKey(W)
    time.sleep(0.015)

"""
In the functions left and right, the amount of time the key S is held corresponds to the magnitude of the steering_angle. This approach ensures that during sharper turns, the car will slow down more so that it doesn't go off the road.
"""

def left(steering_angle):
    ReleaseKey(W)
    ReleaseKey(D)
    PressKey(A)
    if(steering_angle)<-0.2:
        PressKey(S)
        time.sleep(0.002)
        ReleaseKey(S)
    if(steering_angle)<-0.3:
        PressKey(S)
        time.sleep(0.003)
        ReleaseKey(S)
    if(steering_angle)<-0.5:
        PressKey(S)
        time.sleep(0.004)
        ReleaseKey(S)
    PressKey(W)    
    time.sleep(0.006)
    ReleaseKey(A)

def right(steering_angle):
    ReleaseKey(W)
    ReleaseKey(A)
    PressKey(D)
    if(steering_angle)>0.2:
        PressKey(S)
        time.sleep(0.002)
        ReleaseKey(S)
    if(steering_angle)>0.3:
        PressKey(S)
        time.sleep(0.003)
        ReleaseKey(S)
    if(steering_angle)>0.5:
        PressKey(S)
        time.sleep(0.004)
        ReleaseKey(S)
    PressKey(W)
    time.sleep(0.006)
    ReleaseKey(D)

def control_car(steering_angle):
    # Determine the direction based on the steering angle
    # Steering angle is a value between -1 (steering wheel turned all the way to the left) and 1 (steering wheel turned
    # all the way to the right)

    if steering_angle < -0.01:
        left(steering_angle)
    elif steering_angle > 0.01:
        right(steering_angle)
    else:
        forward()

def main():
    # Set the frame rate and time delay (in seconds) for the screen capture
    FPS = 60
    DELAY = 1 / FPS

    # Disable GPU usage for TensorFlow
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    tf.config.set_visible_devices([], 'GPU')

    # Load the pre-trained steering prediction model
    model_keepline = tf.keras.models.load_model('pilotnet_gta5/model_steer.h5', compile=False)

    steering_angle = 0

    # Wait for 3 seconds
    for j in range(3):
        time.sleep(j)

    # Start driving...    
    while True:
        # Capture the screen image
        img = screenshot(region=[0, 40, 800, 600])  # x,y,w,h
        printscreen = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGRA2RGB)
        screen = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (160,120))

        # Make a prediction of steering angle using the model based on the captured screen
        output = model_keepline.predict([screen.reshape(-1,160,120,1)],verbose = 0)

        print(output[0][0])

        # Control the car based on the steering angle
        control_car(output[0][0])

        # Delay the loop to match the frame rate
        time.sleep(DELAY-0.015)


if __name__ == '__main__':
    main()
