
# coding: utf-8

# In[1]:


##import libraries
import cv2
import numpy as np
import math

#load img from pc
img=cv2.imread("ffh.png",0)

cv2.imshow("original",img)
#background Removal
#remove noise
img_blur = cv2.GaussianBlur(img,(3,3),0)
#apply threshold
th, th3 = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY)
cv2.imshow("threshold",th3)
#difine the kernal we will compare to input image matrices
kernal = np.ones((5,5), np.uint8)
#Erosion .. add pixels ro the boundries of object if the kernal "fits"
erosion = cv2.erode(th3,kernal,iterations=2)
cv2.imshow('erosion',erosion)
# Find Canny edges ..used to reduce the noise of the unneccesraly contours
edges = cv2.Canny(img,30,200)
# cv2.imshow('Canny edges',edges)
#Finding contours
edges, contours, hierarchy = cv2.findContours(erosion,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

#Sort contours large to small
sorted_contours= sorted(contours,key=cv2.contourArea,reverse=True)
# cv2.drawContours(img, sorted_contours[0], -1, (255,255,255), 1, cv2.LINE_AA)
# cv2.imshow("Result", img)
#create the mask
mask = np.zeros(img.shape[:2], dtype="uint8") * 255
# Draw the contours on the mask and fill it with color
mask = cv2.fillPoly(mask, pts =[sorted_contours[0]], color=255) 
cv2.imshow("mask",mask)
img = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("object clear", img)
cv2.waitKey()
cv2.destroyAllWindows()


#-----------------fracture detection-------------------

#detect circle bone
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,1,20, param1=4,param2=2,minRadius=38,maxRadius = 70)
for i in circles [:, 0]:
   # draw the outer circle
#     cv2.circle(img,(i[0],i[1]),i[2],(255,255,255),1)
   # draw the center of the circle
    cv2.circle(circles,(i[0],i[1]),2,(255,255,255),1)
    # get rectangle bounding circular bone
xcenter=int(i[0])
ycenter=int(i[1])
r=i[2]
x_start=int(i[0]-i[2])
x_end=int(xcenter+r)
r_point_x=math.floor(i[2]+i[0])
r_point_y=math.floor(-i[2]+i[1])
w=h=math.floor(2.5*i[2])
x=math.floor(i[0]-h)
y=math.floor(i[1]-w)
cv2.imshow("im",img)
cv2.waitKey()

#crop ROI from img
ROI = img[y:y+2*h, x:x+2*w]
cv2.imshow("ROI",ROI)
# apply threshold
equ = cv2.equalizeHist(ROI)
th3 = cv2.adaptiveThreshold(ROI,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow("threshold",th3)

# kernal for closing
kernal = np.ones((5,5), np.uint8)
closing = cv2.morphologyEx(th3 , cv2.MORPH_OPEN ,kernal)
cv2.imshow('Closing',closing)

erosion = cv2.erode(th3,kernal,iterations=1)
cv2.imshow('erosion',erosion)
#detect straight lines on edges
edges = cv2.Canny(closing,50,150,apertureSize = 3)
minLineLength = 5
maxLineGap = 10
arr5=[]
lines = cv2.HoughLinesP(edges,1,np.pi/90,3,minLineLength,maxLineGap)
#check if the bone is straight-in case of negative- or not-in case of fracture-
for i in lines:
    for x1,y1,x2,y2 in i:
        if (x1<(x_end)) and (x1>(x_start)) and (y1<(ycenter)) and (y2<(ycenter)):      
            cv2.line(ROI,(x1,y1),(x2,y2),(255),2)
            arr5.append(i)

if (len(arr5)>0):
    print("negative")
else:
    print("fraction detected")

cv2.imshow("cont",ROI)

cv2.waitKey(0)
cv2.destroyAllWindows()

