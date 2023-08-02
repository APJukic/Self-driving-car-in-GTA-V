"""
Model source: https://github.com/tensorflow/models
License: Apache License 2.0 (https://github.com/tensorflow/models/blob/master/LICENSE)
Description: Script uses pretrained MobileNet SSD model for object detection. The model has been trained to detect various objects belonging to 90 different classes, including vehicles. The script captures the screen and feeds the captured image into the model for object detection. The model predicts information about the detected vehicles, which is used to identify the closest vehicle in proximity to the self-driving car. If the identified vehicle is too close, the self-driving car will automatically slow down, ensuring that there is no collison.
"""
import numpy as np
import os
import sys
from pyautogui import screenshot
import tensorflow as tf
import cv2
import time
import pydirectinput
from utils import label_map_util

# Load a frozen TensorFlow model into memory from a checkpoint file.
def load_detection_graph(path_to_ckpt):
    with tf.io.gfile.GFile(path_to_ckpt, 'rb') as file:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(file.read())
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    return graph

detection_graph = load_detection_graph('ssd_mobilenet_v1_coco/frozen_inference_graph.pb')

# Load the label map for object classes
label_map = label_map_util.load_labelmap('data_coco/mscoco_label_map.pbtxt')


# Start vehicle detection
with detection_graph.as_default():
    with tf.compat.v1.Session(graph = detection_graph, config = tf.compat.v1.ConfigProto()) as sess:
        breaking = False
        while True:
            # Capture screen and prepare for detection
            screen = screenshot(region=[0, 40, 800, 600])
            screen = cv2.cvtColor(np.asarray(screen), cv2.COLOR_BGR2RGB)
            screen = cv2.resize(screen, (800, 450))
            screen = np.expand_dims(screen, axis=0)

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Perform detection:
	    # The model detects objects of different classes and provides information for each detected object, including           
            # its class label (variable classes), confidence score (variable scores), location (variable boxes) and number of detected objects (variable num_detections). 
            # The confidence score represents the probability that the detected object belongs to the predicted class. The object's location consists of two values: 
            # the coordinates of the top-left point and the bottom-right point of the bounding box that encompasses the object like this:
            """

                                 .-.                                    ,-.
                              .-(   )-.                              ,-(   )-.
                            (     __) )-.                        ,-(_      __)
                          `-(       __)                      (_    )  __)-'
                             `(____)-',                        `-(____)-'
                          - -  :   :  - -
                              / `-' \                                        ,@@@@@@@,
                            ,    |   .                               ,,,.   ,@@@@@@/@@,  .oo8888o.
                                 .                                ,&%%&%&&%,@@@@@/@@@@@@,8888\88/8o
             -------------------------------------------         ,%&\%&&%&&%,@@@\@@@/@@@88\88888/88'
                                                         \       %&&%&%&/%&&%@@\@@/ /@@@88888\88888'
              top-left point X__ __ __ __ __ __ __ __     \      %&&%/ %&%%&&@@\ V /@@' `88\8 `/88'  
                             |        _____________   |    \     `&%\ ` /%&'    |.|        \ '|8'
                             |   ____//__][__][___|   |     \        |o|        | |         | |
                             |  (o  _|  -|     _ o|   |      \       |.|        | |         | |
                             |   `-(_)--------(_)-'   |       \      | |        | |         | |
                             | __ __ __ __ __ __ __ __|X       \  \\/ ._\//_/__/  ,\_//__\\/.  \_//__/_
             ------------------------------------------ bottom-right point
            """
 			     
            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                feed_dict={image_tensor: screen})

            # Process vehicle detections
            vehicle_dict = {}
            vehicle_classes = [3, 6, 8] # car, bus, truck

            for i, b in enumerate(boxes[0]):
                if classes[0][i] in vehicle_classes and scores[0][i] >= 0.5: 
                    # Includes only detected objects that are in the class vehicle and who's confidence score is greater than 0.5, 
                    # which means they are relatively close (the closer they are to a self driving vehicle, the greater their confidence score is)

                    mid_x = (b[1] + b[3]) / 2
                    mid_y = (b[0] + b[2]) / 2
                    apx_distance = round(((1 - (b[3] - b[1])) ** 4), 3) # Approximate distance to the detected vehicle
                    vehicle_dict[apx_distance] = [mid_x, mid_y, scores[0][i]]

            # Self driving vehicle control based on detections
            if len(vehicle_dict) > 0:
                # Find approximate distance of the closest vehicle
                closest = sorted(vehicle_dict.keys())[0]

                print(f"Closest vehicle: {vehicle_dict[closest]}")

                # Self driving car slows down if:  
                # distance to the closest vehicle is smaller than threshold (0.4) and the closest vehicle is in front of the self driving car and in the same lane
                if closest < 0.4  and vehicle_dict[closest][0] >0.15 and vehicle_dict[closest][1]<0.82:
                    print("Breaking")
                    pydirectinput.keyDown('space')
                    time.sleep(0.015)
                    breaking = True
                else:
                    breaking = False

                if not breaking:
                    pydirectinput.keyUp('space')
            else:
                breaking = False
                pydirectinput.keyUp('space')

