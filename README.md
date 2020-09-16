# OpenCV-Face-Recognition

1.Basic.py

Step-1:- Import images and read it with the help of cv2. Then it can convert it to RGB because cv2 read the image in the form of BGR but face recognition accepts only RBG.

Step-2:- We need to find out the face loaction with the help of face_recognition.face_locations. Then, We can create encodings for that face with the help of face_recognition.face_encodings

Step-3:- Final step,we need to compare the encoding images with new image with the help of face_recognition.compare_faces

2.Face Recognition + Attendance.py

  We need to follow the above steps.
We can access VideoCapture to find out the know faces and unkonw faces and match it with encoded faces. Then, we can create sample cvs file.in that name and time can be stored automatically if the encoded faces is detected in videocapture.
