import cv2
import numpy as np

frame = cv2.imread ('../Photos/collage.png')
edges = cv2.Canny(frame,50,100) # This uses the canny edge detector. The 100 and 200 are rather arbitrary parameters; the second should be larger than the first, play around to see what numbers work best for each image.

pent = cv2.imread('../Photos/pentagon.png')
pentCanny = cv2.Canny(pent,50,100) #make a canny
pentContours, hierarchy = cv2.findContours(pentCanny,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE) 

pentBlank = np.zeros(pent.shape)
cv2.polylines(pentBlank,pentContours,True,(255),1)
# Find its contours and create a moment set for checking

pentMoments = cv2.moments(pentContours[1]) #moment is average of intensities, which allows us to get the center of a contour
pentHuMoments= cv2.HuMoments(pentMoments)

contours, hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
# Get rid of the ones with an area smaller than tiny

blankImage = np.zeros(edges.shape)

goodContours=[]
for contour in contours:
    if cv2.contourArea(contour)>100:
        contourMoments = cv2.moments(contour)
        contourHuMoments = cv2.HuMoments(contourMoments)
        #find the difference between moments
        delta = np.sum(pentHuMoments-contourHuMoments)
        print(delta)
        if (np.abs(delta)<0.002): #0.002 is our threshold
            print(np.abs(delta))
            goodContours.append(contour)
            cv2.polylines(blankImage,contour,True,(255),1)

cv2.imshow("good contours", blankImage) 
cv2.waitKey(-1)
