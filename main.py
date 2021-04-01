import numpy as np
import face_recognition
import os
import cv2
from datetime import datetime

path = 'Data Set'


def Attendance_Marker(name):
    with open('Attendance.csv', 'r+') as file:
        names = []
        presentData = file.readlines()
        for line in presentData:
            nameInList = line.split(',')
            names.append(nameInList[0])  # All names in the sheet being updated into a list
        if name not in names:
            time = datetime.now()
            time = time.strftime('%H:%M:%S')
            file.writelines(f'\n{name},{time}')


images = []
image_names = []
img_dir = []
encodings = []
main_dir = os.listdir(path)
print(main_dir)


for dir in main_dir:
    for img in os.listdir(f'{path}/{dir}'):
        img_dir.append(img)
print(img_dir)

for dir in main_dir:
    for img in os.listdir(f'{path}/{dir}'):
        image = cv2.imread(f'{path}/{dir}/{img}')
        images.append(image)
        image_names.append(os.path.splitext(img)[0])
print(image_names)

for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings.append(face_recognition.face_encodings(img)[0])

cam = cv2.VideoCapture(1)

while True:
    success, img = cam.read()
    # imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    imagesInFrame = face_recognition.face_locations(imgS)
    encodingsInFrame = face_recognition.face_encodings(imgS, imagesInFrame)

    for code, loc in zip(encodingsInFrame, imagesInFrame):
        result = face_recognition.compare_faces(encodings, code)
        distance = face_recognition.face_distance(encodings, code)
        print(distance)
        index = np.argmin(distance)

        if result[index]:
            match_name = image_names[int(index)]
            match_dir = ""
            y1, x2, y2, x1 = loc
            # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 15), (x2, y2), (0, 0, 255), cv2.FILLED)
            for i in main_dir:
                for j in os.listdir(f'{path}/{i}'):
                    if match_name == os.path.splitext(j)[0]:
                        match_dir = i
            print(match_dir)
            cv2.putText(img, match_dir, (x1 + 6, y2 - 4), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
            Attendance_Marker(match_dir)
    cv2.imshow('Found', img)
    cv2.waitKey(0)
