About:
-
Self driving car in GTA V using PilotNet and MobileNet SSD.
PilotNet is employed for lane-keeping, ensuring the car stays on the road, while MobileNet SSD is utilized for detecting nearby cars. 
This enables the self-driving car to brake when it approaches other vehicles in close proximity.

It is recommended to use speed limiter since this project doesn't contain throttle prediction. When crossing the intersection user should correct the vehicle since turning at the intersection is not implemented.

Files:
-
-drive.py enables autonomous driving for the car using PilotNet. The model predicts the car's steering angle by analyzing the captured screen. Based on these predictions, the script navigates the car to turn left, right, or move straight ahead, ensuring that the car is able to stay in its lane while driving.

-detect.py uses MobileNet SSD model for object detection. The script captures the screen and feeds the captured image into the model for object detection. The model predicts information about the detected vehicles, which is used to identify the closest vehicle in proximity to the self-driving car. If the identified vehicle is too close, the self-driving car will automatically slow down, ensuring that there is no collison.

-pilotnet_training.ipynb notebook is used for training PilotNet

-pilotnet_gta5 contains trained PilotNet Model

-ssd_mobilenet_v1_coco contains pretrained MobileNest SSD model downloaded from: https://github.com/tensorflow/models

Requirements:
- 
1.GTA 5 mods:

    hood camera: https://www.gta5-mods.com/scripts/hood-camera
    
    Enhanced Native Trainer: https://www.gta5-mods.com/scripts/enhanced-native-trainer-zemanez-and-others
    
2.Python 3.6

3.Tensorflow

4.Keras

Usage:
-
1. Run GTA 5 in windowed mode with resolution 800x600. Drag the window to left top corner of the screen
2. With Enhanced Native Trainer limit the speed (30 mph)
3. Enter the car and turn on hood camera
4. Run drive.py
5. Run detect.py

Demo:
-
https://www.youtube.com/watch?v=SAk1ZFs-va0&t=9s
