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
            imageWrite() --> Save the output in system
            imageDisplay() --> Display the output
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


class ShiTomasiCornerDetection:

    """
        Responsibility :  To find the corners in an image using Harris Corner Detection Algorithm 
        Functions :
            __init__() --> Initialize Filter , weights, k and threshold values
            shiResponseScoreMatrix() --> Function to compute the harris corner score for each pixel neighbourhood
            cornerDistance() --. Comnpute Distance between two corners
            goodFeatures() --> to select the best features out of all
            nonMaximumSuppression() --> Function to remove false positives
            visualizeCornerPoints() --> Function to draw circle on the detected corners in 

    """

    def __init__(self):
        self.Sx = (1/8) * np.asarray([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        self.Sy = (1/8) * np.asarray([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
        self.w =  np.asarray([[1,1,1],[1,1,1],[1,1,1]]) 
        self.threshold = 1000

    def shiResponseScoreMatrix(self, gray_image):
        """
            @args gray_image --> Input Grayscale Image
            @return M --> Shi Tomasi Corner Response Matrix
        """
        Ix = signal.convolve2d(gray_image, self.Sx, boundary='symm', mode='same')
        Iy = signal.convolve2d(gray_image, self.Sy, boundary='symm', mode='same')
        Ixx = Ix * Ix 
        Ixy = Ix * Iy 
        Iyy = Iy * Iy 
        Sxx = signal.convolve2d(Ixx, self.w, boundary='symm', mode='same')
        Sxy = signal.convolve2d(Ixy, self.w, boundary='symm', mode='same')
        Syy = signal.convolve2d(Iyy, self.w, boundary='symm', mode='same')
        M = np.zeros((gray_image.shape[0], gray_image.shape[1]))
        for i in range(gray_image.shape[0]):
            for j in range(gray_image.shape[1]):
                w, v = np.linalg.eig(np.asarray([[Sxx[i,j], Sxy[i,j]],[Sxy[i,j], Syy[i,j]]]))
                M[i,j] = np.min(w)
        M[M < self.threshold] = 0
        return M

    def nonMaximumSuppression(self, M, m, n):
        """
            @args : M --> Shi tomasi Corner Response matrix
            @args : m,n --> Shape of matrix
            @return : M --> shi tomasi Corner Response Matrix after removing false possitives
        """
        for r in range(1,m-1):
            for c in range(1, n-1):
                patch = M[r-1:r+2, c-1:c+2]
                if (np.max(patch) != M[r,c]):
                    M[r,c] = 0
        return M

    def cornerDistance(self, x1, x2):
        """
            @args: x1, x2 --> Coordinates
            @return Eucledian Distance
        """
        return np.hypot(x1[0]-x2[0],x1[1]-x2[1])

    def goodFeatures(self, M, max_features=500, min_distance = 10):
        """
            @args: M --> Shi tomasi Corner Response matrix
            @args: max_features --> No of corner points required
            @min_distance --> Distance between the features
            @return --> List containing best corner points
        """
        corners = []
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if(M[i,j] > 0):
                    corners.append((i,j))
        sorted_corners =sorted(corners, key=lambda k:-M[k])
        best_corners = [sorted_corners.pop(0)]
        for c in sorted_corners:
            dst = [self.cornerDistance(c,bc) for bc in best_corners]
            if max(dst) >= min_distance:
                best_corners.append(c)
            if len(best_corners) == max_features:
                break
        return best_corners

    def visualizeCornerPoints(self, kps, bgr_image):
        """
            @args: kps --> list contatining best corner points in the image
            @args: bgr_image --> Input RGB image
            @return --> RGB image marked with corner points
            
        """
        for p in kps:
            cv2.circle(bgr_image,(p[1],p[0]), color=(0, 0, 255), radius = 3)
        return bgr_image
