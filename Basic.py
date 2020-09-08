import cv2
import numpy as np
import face_recognition

### Step-1
"""Johnny-1-Known Face
Johnny-3-Test Face
Johnny-2 & Robert-Comparing Face"""


"""Image import and convert to RGB"""
imgJohnny1 = face_recognition.load_image_file("Basic Images/Johnny-1.jpg") # it helps to read the images
#imgJohnny1 = cv2.resize(imgJohnny1,(1000,1000))
imgJohnny1 = cv2.cvtColor(imgJohnny1,cv2.COLOR_BGR2RGB) # CV2 image has BGR but face recognition accepts only RGB.Se we need to convert

imgJohnny2 = face_recognition.load_image_file("Basic Images/Johnny-2.jpg") # it helps to read the images
#imgJohnny2 = cv2.resize(imgJohnny2,(1000,1000))
imgJohnny2 = cv2.cvtColor(imgJohnny2,cv2.COLOR_BGR2RGB) # CV2 image has BGR but face recognition accepts only RGB.Se we need to convert

imgJohnny3 = face_recognition.load_image_file("Basic Images/Johnny-3.jpg") # it helps to read the images
#imgJohnny3 = cv2.resize(imgJohnny3,(1000,1000))
imgJohnny3 = cv2.cvtColor(imgJohnny3,cv2.COLOR_BGR2RGB) # CV2 image has BGR but face recognition accepts only RGB.Se we need to convert

imgRobert = face_recognition.load_image_file("Basic Images/Robert.jpg")
imgRobert = cv2.cvtColor(imgRobert,cv2.COLOR_BGR2RGB)

### Step-2

"""Finding theire faces in images and encodings as well """

faceLoc1 = face_recognition.face_locations(imgJohnny1)[0] # Helps to find out the face in image
#print(faceLoc)
encodeJohnny1 = face_recognition.face_encodings(imgJohnny1)[0] # Helps to encode the face in image
cv2.rectangle(imgJohnny1,(faceLoc1[3],faceLoc1[0]),(faceLoc1[1],faceLoc1[2]),(255,0,255),3) # It helps to draw rectangle to image face

"""Same thing we can do for our second image which is test image & with this example we did not need face location """

faceLoc3 = face_recognition.face_locations(imgJohnny3)[0]
encodeJohnny3 = face_recognition.face_encodings(imgJohnny2)[0] # Helps to encode the face in image
cv2.rectangle(imgJohnny3,(faceLoc3[3],faceLoc3[0]),(faceLoc3[1],faceLoc3[2]),(255,0,255),3) # It helps to draw rectangle to image face

### Encoding for Comparing Image
faceLoc4 = face_recognition.face_locations(imgRobert)[0]
encodeRobert = face_recognition.face_encodings(imgRobert)[0]
cv2.rectangle(imgRobert,(faceLoc4[3],faceLoc4[0]),(faceLoc4[1],faceLoc4[2]),(255,0,255),3)

### Step-3 Final Step
"""Compare these faces and finding distance between them"""

results1 = face_recognition.compare_faces([encodeJohnny1],encodeJohnny3) #encodeJohnny1-Main image,encodeJohnny3-Comparing Image
results2 = face_recognition.compare_faces([encodeJohnny1],encodeRobert) #encodeJohnny1-Main image,encodeRobert-Comparing Image
print(results1)
print(results2)
"""If result is true,both images are same person otherwise different persons"""
"""Sometimes we have lot of images there can be similarities.So,we find out how similar these images are.
for that we need to find the distance"""
faceDist1 = face_recognition.face_distance([encodeJohnny1],encodeJohnny3)
print(results1,faceDist1)

faceDist2 = face_recognition.face_distance([encodeJohnny1],encodeRobert)
print(results2,faceDist2)
cv2.putText(imgJohnny3,f"{results1}{round(faceDist1[0],2)}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.putText(imgRobert,f"{results2}{round(faceDist2[0],2)}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow("Jack Sparow",imgJohnny1)
cv2.imshow("Johnny Depp",imgJohnny2)
cv2.imshow("Johnny",imgJohnny3)
cv2.imshow("Robert",imgRobert)

cv2.waitKey(0)