import numpy as np
import cv2
import os

img = cv2.imread(r'../Faces/usrc_cropped.png')

# Loop through both files, cropped and uncropped
for i in range(2):
    if i == 0:
        img = cv2.imread(r'../Faces/usrc_all.png')
        input_type = 'All USRC'

    else:
        img = cv2.imread(r'../Faces/usrc_cropped.png')
        input_type = 'Cropped USRC'

    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    pic_frame = cv2.imread(r'../Faces/imageframe.png')

    haar_cascade = cv2.CascadeClassifier('haar_face.xml')
    faces_rect = haar_cascade.detectMultiScale(grey,scaleFactor = 1.1, minNeighbors = 10)
    
    # Initialise k counter to update file name
    k=0
    for (x,y,w,h) in faces_rect:


        faceROI = img[y:y+h,x:x+w,:]
        thisPicFrame = np.zeros((w+40, h+40, 3))
        thisPicFrame = cv2.copyTo(pic_frame,None)
        thisPicFrame = cv2.resize(thisPicFrame, (w+40,h+40))
        thisPicFrame[20:h+20,20:w+20,:] = faceROI

        #Initialise output path based on file input and no. of face detected
        output_path = '../Faces/' + input_type + '-' + str(k) +'.png'
        cv2.imwrite(output_path,thisPicFrame)
        cv2.imshow(input_type,thisPicFrame)

        #Update counter and prompt user to continue running
        k += 1
        cv2.waitKey(-1)