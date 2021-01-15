# Import Neccesary Packages
import numpy as np 
import cv2
from matplotlib import pyplot as plt
import copy
import math

class Util:


    """
        Responsibilty : Read, Write, Display, splitRGB, Expand
        Functions:
            imageRead() --> Reads the image from the system and gives us an color image to perform operation
            imageWrite() --> Stores the image in the system
            imageDisplay() --> Displays the image to the user
            splitBGR() --> splits a 3 channel RGB image into 3 single channel images
            mergeBGR() --> joins 3 images of R,G,B channel to give a single color Image
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

    def splitBGR(self,image):
        """
            @args: RGB image
            @return : B,G,R channel images

        """
        return cv2.split(image)

    def mergeBGR(self,b,g,r):
        """
            @args : b,g,r channel images
            @return : rgb image
        """
        return cv2.merge((b,g,r))

    def padImage(self,org_image,filter_size):
        """
            @args: original image and size of the filter to be operated on the original image
            @return : Expanded image
        """
        pad_size = int(filter_size/2)
        return cv2.copyMakeBorder( org_image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)

class BilateralFilter:

    """
        Responsibility : To perform Bilateral Filtering on Images to smooth out the image while preserving the edge
        Functions:
            l2Norm() --> To fine the eucledian Distance between two pixel locations
            gaussian() --> returns gaussian value for corresponding value and Standard deviation
            own_implementation() --> Applies Bilateral Filtering from scratch
            opencv_implementation --> Applies opencv functions for Bilateral Filtering
    """

    def l2Norm(self,x, y, i, j):
        
        """
            @args : pixel locations of neighbours and current pixel
            @return : Eucledian distance for between given locations
        """
        return np.sqrt((x-i)**2 + (y-j)**2)

    def gaussian(self,val, sigma):
        
        """
            @args: value and standard deviation
            @return : Corresponding gaussian value
        """
        return (1.0 / ( (2 * math.pi * (sigma ** 2)) ** 0.5)) * math.exp(- (val ** 2) / (2 * sigma ** 2))

    def own_implementation(self,org_image,padded_image,filter_size,sigma_i, sigma_s):

        """
            @args :
                org_image : original unpadded image
                padded_image: padded image for solving boundary cases
                filer_size : size of filter to be used
                sigma_i :  standard deviation for Intensity Values
                sigma_s :  standard deviation for spatial values

            @return : smoothened image with same size of original image
        """
        filtered_image = np.zeros_like(org_image)
        h,w = org_image.shape 
        mid = int(filter_size/2)
        for i in range(h):
            for j in range(w):
                filter_starti = i 
                filter_endi = i + filter_size
                filter_startj = j  
                filter_endj = j + filter_size
                val = 0
                const = 0
                for ni in  range(filter_starti, filter_endi):
                    for nj in range(filter_startj, filter_endj):
                        neighbour_i = i - mid + ni 
                        neighbour_j = j - mid + nj
                        gi = self.gaussian(padded_image[ni,nj] - org_image[i,j], sigma_i)
                        gs = self.gaussian(self.l2Norm(neighbour_i, neighbour_j, i, j), sigma_s)
                        w0 = gi * gs
                        val += padded_image[ni,nj] * w0
                        const += w0 
                val = val / const
                filtered_image[i,j] = int(val)

        return filtered_image

    def opencv_implementation(self,org_image,filter_size,sigma_i,sigma_s):
        
        """
            @args : originalImage, filterSIze, spatial and color intensity standard deviation
            @return : smoothened image with same size as original image
        """
        return cv2.bilateralFilter(org_image, filter_size, sigma_i, sigma_s)