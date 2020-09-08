import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

## Step-1
"""Import Images & Converted Into RGB"""
path = 'Images'
images = [] # List of all images
names = [] # Image names
mylist = os.listdir(path) # Grab the list of images from this folder
#print(mylist)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}') #cls is name of the image
    images.append(curImg)
    names.append(os.path.splitext(cl)[0])
print(names) # With out extentions of name(i.e eliminate .jpg))

## Step-2
"""Convert BGR To RGB & Encodings For Each Image"""

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #imgFace = face_recognition.face_locations(img)[0]
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print(len(encodeListKnown))

## Step-3
"""Access Webcam To Match It With Stored Images"""

cap = cv2.VideoCapture(-1)
while True: # To get each frame one by one
    success,img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25) # (0,0)-Pixle size,0.25&0.25-Scale
    """Resize-In realtime it takes more time to run.So, we can reduce the size of the image"""
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    """In web came we find multiple faces,So, We find the face location and encode it"""
    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace) # Match the face with list face and webcam face
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace) # Calculate distance of know face and webcam face(low distance gives better result)
        #print(faceDis)
        matcheIndex = np.argmax(faceDis)

        if matches[matcheIndex]:
            name = names[matcheIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

