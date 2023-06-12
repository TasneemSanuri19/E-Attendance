from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.http import FileResponse
import cv2
import glob
import os
from math import floor
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import csv
import shutil
import pandas as pd
import subprocess
import ssl

num_to_name = {'abdelnasser abuziuter': 0, 'abdelrahman elayyan': 1, 'ahmad ghuneim': 2, 'aya ladadweh': 3,
               'aya sulaq': 4,
               'ban qaqish': 5, 'bana hjeji': 6, 'batool barakat': 7, 'christine amareen': 8, 'dana twal ': 9,
               'fahd othman': 10,
               'hashem thabsem': 11, 'jana shaer': 12, 'kholoud qubbaj': 13, 'lara diab': 14, 'malak abdelwahab': 15,
               'mohammad jumaa': 16,
               'noor alawi': 17, 'noor awwad': 18, 'osama zamel': 19, 'raghad abu tarboush': 20, 'rakan armoush': 21,
               'rama al alawneh': 22,
               'raneem nabulsea': 23, 'reem assi': 24, 'rose al nairab': 25, 'saif aburaisi': 26, 'saja taweel': 27,
               'samira abubakr': 28,
               'sanad abu khalaf': 29, 'sara darwish': 30, 'sara selwadi': 31, 'shahem al naber': 32,
               'suhaib abu kiwan': 33, 'suheil wakileh': 34,
               'tamara kawamleh': 35, 'tareq sarayji': 36, 'tariq sallam': 37, 'tasneem sanuri': 38,
               'waleed abujaish': 39, 'yara matarneh': 40,
               'yousef shaqadan': 41, 'zain shaarawi': 42}
inverted_dict = {v: k for k, v in num_to_name.items()}
students = num_to_name.keys()


def e_attendance_home_screen(request):
    return render(request, 'eAttendanceHomeScreen.html')


def empty_directory(directory_path):
    # Iterate over all the files and subdirectories in the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Construct the full file path
            file_path = os.path.join(root, file)
            # Remove the file
            os.remove(file_path)
        for dir in dirs:
            # Construct the full directory path
            dir_path = os.path.join(root, dir)
            # Remove the directory
            os.rmdir(dir_path)


@api_view(['POST'])
def upload_video_one(request):
    # Check if the request contains a file
    video_file = request.data['file'].temporary_file_path()
    current_dir = os.getcwd()

    # Join the current directory and the file name to create the file path
    filepath = os.path.join(current_dir, video_file)

    extract_frames_for_video_one(filepath)
    return JsonResponse({'message': 'Video uploaded and processed successfully'})


def extract_frames_for_video_one(video1_path):
    video = cv2.VideoCapture(video1_path)
    # Count the number of frames and FPS
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    seconds = floor(frames / fps)
    mins = floor(seconds / 60)
    minutes = 1
    seconds = 0
    while minutes <= mins:
        frame_id = int(fps * (minutes * 60 + seconds))
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = video.read()
        # Here put the path of the frames folder you created in your drive, and each time change the name of the video according to the video you have
        cv2.imwrite(r'frames\\vid1frame{}.png'.format(frame_id), frame)
        minutes += 1


@api_view(['POST'])
def upload_video_two_and_check_attendance(request):
    video_file = request.data['file'].temporary_file_path()
    current_dir = os.getcwd()

    # Join the current directory and the file name to create the file path
    filepath = os.path.join(current_dir, video_file)
    # Print the file path
    attendance_sheet = []
    video = cv2.VideoCapture(filepath)
    # Count the number of frames and FPS
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    seconds = floor(frames / fps)
    mins = floor(seconds / 60)
    minutes = 1
    seconds = 0
    while minutes <= mins:
        frame_id = int(fps * (minutes * 60 + seconds))
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = video.read()
        # Here put the path of the frames folder you created in your drive, and each time change
        # the name of the video according to the video you have
        cv2.imwrite(r'frames\\vid2frame{}.png'.format(frame_id), frame)
        minutes += 1

    # Detection

    subprocess.run(
        "python yolov5\\detect.py --weights Detection_Weights.pt --img 640 --conf 0.4 --source frames\\*.png --save-crop",
        shell=True)

    data_dir = "yolov5\\runs\\detect\\exp\\crops\\face"

    saved_weights_path = 'Recognition_Weights_224 (2).pt'

    def predict_labels(image_dir, saved_weights_path):
        ssl._create_default_https_context = ssl._create_unverified_context
        model = models.resnet18(pretrained=True)

        # Modify the fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 43)

        saved_weights = torch.load(saved_weights_path, map_location=torch.device('cpu'))

        device = torch.device('cpu')

        model.load_state_dict(saved_weights)

        # Move the model to the selected device
        model = model.to(device)

        # Switch to evaluation mode
        model.eval()

        # Transformations
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Get a list of all image file paths in the directory
        image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))

        # Create a list to store the predicted labels
        predicted_labels = []
        for path in image_paths:
            image = Image.open(path)
            image = transform(image).unsqueeze(0)
            with torch.no_grad():
                image = image.to(device)
                output = model(image)
                _, predicted = torch.max(output.data, 1)
                predicted_labels.append(predicted.item())

        result = []
        for i in predicted_labels:
            if predicted_labels.count(i) >= 1 and i not in result:
                result.append(i)
        result_names = [inverted_dict[num] for num in result]
        attendance_dict = {student: 1 if student in result_names else 0 for student in students}
        with open('attendance.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Attendance'])
            for student in students:
                writer.writerow([student, attendance_dict[student]])
        frames_dir = 'frames'
        empty_directory(frames_dir)
        shutil.rmtree("yolov5/runs")
        return 'attendance.csv'

    attendance_file_path = predict_labels(data_dir, saved_weights_path)
    return Response({'attendance_file_path': attendance_file_path})


@api_view(['GET'])
def send_excel_file_and_render_result_page(request):
    file_path = request.GET.get('file_path', '')
    context = {'excel_file_path': file_path}
    return render(request, 'resultExcelPage.html', context)


@api_view(['GET'])
def download_excel(request):
    # Get the file path from the query parameters or request body if necessary
    excel_file_path = request.GET.get('file_path', '')

    # Send the file as a response for download
    file_name = excel_file_path.split('/')[-1]
    response = FileResponse(open(excel_file_path, 'rb'))
    response['Content-Disposition'] = f'attachment; filename="{file_name}"'
    return response
