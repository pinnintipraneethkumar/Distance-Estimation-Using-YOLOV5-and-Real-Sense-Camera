# Food-Object-Detection-and-Distance-Estimation-Using-YOLOV5-and-Real-Sense-Camera
## Project Overview :

The aim of this project is to develop a system for food object detection and distance estimation using the YOLOv5 model in conjunction with a Real Sense camera. The project focuses on addressing the need for efficient and accurate food detection, which has various applications in areas such as inventory management, food quality control, and dietary monitoring. By leveraging the YOLOv5 model, the system will be capable of detecting various food objects in real time, while the integration with the Real Sense camera will enable accurate estimation of the distance between the camera and the detected food objects.

### __Pycache__ : The Pycache folder contains the real-sense camera library.

### Config_files : 

The config_files folder contains the classes1.txt file. you can edit the classes1.txt file to create your own custom classes or create a new classes.txt file and you can download our [trained model best.onnx file](https://drive.google.com/file/d/1GdW4rMaqCUFyjkiP_yno7nKXrJYHQjej/view?usp=sharing)

### python : 

The Python folder contains the two .py files. realsense_camera.py file contains the code required to access the real-sense camera and get the real and depth image frames. The frames are fed into the trained model (best.onnx) in yolo.py file.


## Usage : 
1.  Clone this repository.


3. Install real- sense camera library.
   '''
   `pip install pyrealsense2`
   '''

4. Run :''' your path/python/yolo.py''' in the command prompt to find the object and the distance.


## Demo : 

https://github.com/pinnintipraneethkumar/Food-Object-Detection-and-Distance-Estimation-Using-YOLOV5-and-Real-Sense-Camera/assets/76033282/bf1982a7-7c34-42d1-8f69-84ca1b65f979





