# Import Neccesary Packages
import numpy as np 
import cv2
from matplotlib import pyplot as plt
import copy
import math
from scipy import signal

class Util:


    """
        Responsibilty : Read, Write, Display, splitRGB, Expand
        Functions:
            imageReadBGR() --> reads the image from the systema and returns a multi dimensional array (3 channel array)
            imageReadGray() --> Reads the image from the system and gives us an gray scale image array to perform operation
            imageStoreDisplay() --> Displays the output and stores the Reslut
            padImage() --> Increases the size of Image to perform operation
    """

    def imageReadBGR(self,fileName):
        """
            @args : fileName (String)
            @return : BGR image (3 channel 2d array)
        """
        image = cv2.imread(fileName)
        if image is not None:
            return image

    def imageReadGray(self,fileName):
        """
            @args : fileName (String)
            @return : grayscale image (1 channel 2d array)
        """
        gray_image = cv2.imread(fileName,0)
        if gray_image is not None:
            return gray_image
        
    def imageWrite(self,image, fileName):
        """
            @args : output image and Name of the file where the output image is to be stored
        """
        cv2.imwrite(fileName,image)
    
    def imageDisplay(self,image, imageName):
        """
            @args : Image to be displayed and the name of the image
        """
        cv2.imshow(imageName,image)
        cv2.waitKey(0)


class HarrisCornerDetection:

    """
        Responsibility :  To find the corners in an image using Harris Corner Detection Algorithm 
        Functions :
            __init__() --> Initialize Filter , weights, k and threshold values
            harrisResponseScoreMatrix() --> Function to compute the harris corner score for each pixel neighbourhood
            nonMaximumSuppression() --> Function to remove false positives
            visualizeCornerPoints() --> Function to draw circle on the detected corners in 

    """

    def __init__(self):
        self.Sx = (1/8) * np.asarray([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        self.Sy = (1/8) * np.asarray([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
        self.w =  np.asarray([[1,1,1],[1,1,1],[1,1,1]]) 
        self.k = 0.06
        self.threshold = 30000000

    def harrisResponseScoreMatrix(self, gray_image):
        Ix = signal.convolve2d(gray_image, self.Sx, boundary='symm', mode='same')
        Iy = signal.convolve2d(gray_image, self.Sy, boundary='symm', mode='same')
        Ixx = Ix * Ix 
        Ixy = Ix * Iy 
        Iyy = Iy * Iy 
        Sxx = signal.convolve2d(Ixx, self.w, boundary='symm', mode='same')
        Sxy = signal.convolve2d(Ixy, self.w, boundary='symm', mode='same')
        Syy = signal.convolve2d(Iyy, self.w, boundary='symm', mode='same')
        det = (Sxx * Syy) - (Sxy * 2)
        trace = Sxx + Syy
        M = det - self.k*(trace ** 2)
        M[M < self.threshold] = 0
        return M

    def nonMaximumSuppression(self, M, m, n):
        for r in range(1,m-1):
            for c in range(1, n-1):
                patch = M[r-1:r+2, c-1:c+2]
                if (np.max(patch) != M[r,c]):
                    M[r,c] = 0
        return M

    def visualizeCornerPoints(self, M, bgr_image, t=1):
        if t == 1:
            kps = np.argwhere(M >0)
        else:
            kps = np.argwhere(M > 0.015 * np.max(M))
        for p in kps:
            cv2.circle(bgr_image,(p[1],p[0]), color=(0, 0, 255), radius = 3)
        return bgr_image