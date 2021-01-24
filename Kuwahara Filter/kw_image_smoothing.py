# Import Neccesary Packages
import numpy as np 
import cv2
from matplotlib import pyplot as plt
import copy
import math

class Util:


    """
        Responsibilty : Read, Write, Display, splitHSV, mergeHSV, Expand
        Functions:
            imageRead() --> Reads the image from the system and gives us an color image to perform operation
            imageWrite() --> Stores the image in the system
            imageDisplay() --> Displays the image to the user
            splitHSV() --> splits a 3 channel RGB image into 3 single channel H,S,V
            mergeHSV() --> joins 3 Channels H,S,V to give a single HSV Image
            toHSV() --> Converts RGB Image to HSV
            toBGR() --> Converts HSV image back to BGR
            padImage() --> Increases the size of Image to perform operation
    """
    
    def imageRead(self,fileName):
        """
            @args : fileName (String)
            @return : grayscale image (1 channel 2d array)
        """
        bgr_image = cv2.imread(fileName)
        if bgr_image is not None:
            return bgr_image
    
    def imageWrite(self, fileName, output):
        """
            @args : output image and Name of the file where the output image is to be stored
        """
        cv2.imwrite(fileName,output)
    
    def imageDisplay(self,imageName, image):
        """
            @args : Image to be displayed and the name of the image
        """
        cv2.imshow(imageName,image)
        cv2.waitKey(0)

    def splitHSV(self,image):
        """
            @args: HSV image
            @return : H,S,V channel images

        """
        return cv2.split(image)

    def mergeHSV(self,h,s,v):
        """
            @args : H,S,V channel images
            @return : HSV image
        """
        return cv2.merge((h,s,v))

    def toHSV(self, rgb_image):
        """
            @args : rgb image
            @return : hsv image
        """
        return cv2.cvtColor(rgb_image,cv2.COLOR_BGR2HSV)

    def toBGR(self, hsv_image):
        """
            @args : hsv image
            @return : rgb image
        """
        return cv2.cvtColor(hsv_image,cv2.COLOR_HSV2BGR)

    def padImage(self,org_image,filter_size):
        """
            @args: original image and size of the filter to be operated on the original image
            @return : Expanded image
        """
        pad_size = int(filter_size/2)
        return cv2.copyMakeBorder( org_image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)

class KuwaharaFilter:
    """
        Scratch Implementation of Kuwahara Filter
    """
    def implementation(self,org_image,padded_image,filter_size):
        filtered_image = np.zeros_like(org_image)
        height,width = org_image.shape 
        mid = int(filter_size/2)
        for i in range(height):
            for j in range(width):
                avg = []
                var = []
                
                q1 = padded_image[i:i+mid+1,j+mid:j+filter_size]
                q1 = q1.flatten()
                q1_mean = np.mean(q1)
                avg.append(q1_mean)
                q1_var = np.var(q1)
                var.append(q1_var)
                q2 = padded_image[i:i+mid+1,j:j+mid+1]
                q2 = q2.flatten()
                q2_mean = np.mean(q2)
                avg.append(q2_mean)
                q2_var = np.var(q2)
                var.append(q2_var)
                q3 = padded_image[i+mid:i+filter_size,j:j+mid+1]
                q3 = q3.flatten()
                q3_mean = np.mean(q3)
                avg.append(q3_mean)
                q3_var = np.var(q3)
                var.append(q3_var)
                q4 = padded_image[i+mid:i+filter_size,j+mid:j+filter_size]
                q4 = q4.flatten()
                q4_mean = np.mean(q4)
                avg.append(q4_mean)
                q4_var = np.var(q4)
                var.append(q4_var)

                loc = var.index(min(var))
                filtered_image[i,j] = int(avg[loc])
        return filtered_image