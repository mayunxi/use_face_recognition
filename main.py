#coding=utf-8
#算法来源：
from PIL import Image
import face_recognition
import time
import cv2
import cv

cam = "rtsp://192.168.1.10:554/user=admin&password=&channel=1&stream=0.sdp?"
print cam
cap = cv2.VideoCapture(cam)

known_image = face_recognition.load_image_file("test.jpg")
cv2.imshow("known",known_image)
print(type(known_image))
yunxi_encoding = face_recognition.face_encodings(known_image)[0]

while(1):
    # Load the jpg file into a numpy array
    # image = face_recognition.load_image_file("/home/stfz/1.jpg") # Load the jpg file into a numpy array

    ret, tempimg = cap.read()  # show a frame

    #print type(tempimg)
    h,w=tempimg.shape[:2]  #提取高和宽
    small_image = cv2.resize(tempimg, (int(0.3*w), int(0.3*h)), interpolation=cv2.INTER_CUBIC)    #缩小尺寸
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_image[:, :, ::-1]


    if ret:


        # Find all the faces in the image using the default HOG-based model.
        # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
        # See also: find_faces_in_picture_cnn.py
        start = time.clock()
        face_locations = face_recognition.face_locations(rgb_small_frame)         #识别人脸

        # unknown_encoding = face_recognition.face_encodings(rgb_small_frame,face_locations)
        # if len(unknown_encoding):
        #     results = face_recognition.face_distance(yunxi_encoding, unknown_encoding)
        #     sim = 1.0 / (1.0 + results)  # 归一化
        #     print("score=", sim[0])

        elapsed = (time.clock() - start)
        print("Time used:",elapsed)

        for face_location in face_locations:
            print("I found {} face(s) in this photograph.".format(len(face_locations)))
            # Print the location of each face in this image
            top, right, bottom, left = face_location
            #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
            cv2.rectangle(small_image,(left-2,top-2),(right+2,bottom+2),(0, 255, 0),2)        #标记人脸位置
        cv2.imshow('frame', small_image)
        cv2.waitKey(1)
