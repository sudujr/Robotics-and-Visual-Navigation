# Import Neccesary Packages
import numpy as np 
import cv2
from matplotlib import pyplot as plt
import copy

#Load the given image
givenImage = cv2.imread('img_3327_o.jpg')
gray = cv2.cvtColor(givenImage, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8)
img_erosion = cv2.erode(thresh1,kernel,iterations=1)
img_dilation = cv2.dilate(thresh1,kernel,iterations=1)
opening = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel)
closing = cv2.morphologyEx(thresh1,cv2.MORPH_CLOSE,kernel)
cv2.imshow('G',thresh1)
cv2.waitKey(0)
cv2.imshow('G',img_erosion)
cv2.waitKey(0)
cv2.imshow('G',img_dilation)
cv2.waitKey(0)
cv2.imshow('G',opening)
cv2.waitKey(0)
cv2.imshow('G',closing)
cv2.waitKey(0)