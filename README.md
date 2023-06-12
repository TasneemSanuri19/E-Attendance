# E-Attendance using face recognition.
This project aims to perform E-attendance through the use of deep learning techniques. The main components of the system are face detection using YOLOv5, and face recognition through ResNet18.

## General Overview of the project
The first step is obtaining the data and feeding it to a YOLOv5 model for face detection. Afterward, images of students’ faces were cropped and separated into corresponding folders. Folders of cropped images would be the training data of the recognition model. The result of the mentioned DL models is then processed and connected to an attendance system to produce the final attendance list.

## Getting Started
```commandline
pip install ultralytics
```


### How to directly use our system - User manual 
<img width="600" alt="Screenshot 2023-06-09 at 7 28 50 PM" src="https://github.com/TasneemSanuri19/E-Attendance/assets/95653786/5d27cb3c-b40b-434b-ac84-0cea2382ce62">


### How to use our code
-	Install the requirements needed to implement this project on Google Colaboratory using python programming language, from “requirements.txt”.
-	clone our repository: git clone https://github.com/aakashjhawar/face-recsing-opencv**ours
-	Use “deployment.py” notebook which works as follows:
     - Accepts the paths of two videos.
     - Applies frame extraction and stores the frames.
     - Feeds the resulting frames into the yolov5 model, where faces are detected and cropped.
     - The cropped images are then inputted into the ResNet18 recognition model.
     - Finally, an attendance list that contains the recognized students will be returned.


### How to train our models on your own data
-	Install the requirements needed to implement this project on Google Colaboratory using python programming language, from “requirements.txt”.
-	clone our repository: git clone https://github.com/aakashjhawar/face-recsing-opencv**ours
-	Perform frame extraction through “frame_extraction.py”.
-	Train your yolov5 detection model on your own data by following the “detection.py” notebook and obtaining the best weights to be used later.
-	Afterwards, train the ResNet18 recognition model, using the “recognition.py” notebook and obtain the best weights to be used later.
-	To finally test your model, you can use the “deployment.py”, but insert your weights instead.


