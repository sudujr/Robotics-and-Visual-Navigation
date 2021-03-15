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
            imageWrite() --> Stores the output image in the system
            imageDisplay() --> View the output image

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

class MoravecCornerDetection:

    """
        Responsibilty : Perform Moravec Corner Detection and Display the Corner points on the image
        Functions:
            __init__() --> initilaize the the shift coordinates (4 discrete shifts)
            neighbourPatch() --> Function to compute the neighbourhood patch based on the shifts
            SSD() --> Function to Compute Sum of Squared Differences
            morvecCornerDetection() --> Function to compute the Q matrix (moravec Corner points are deduced from the Q matrix)
            nonMaximumSuppression() --> Function to remove false positves
            visualizeCornerPoints() --> Function to visualize the cprner points based on the threshold


    """

    def __init__(self):
        self.shifts = [[1,0],[0,1],[1,1],[-1,1]]

    def neighbourPatch(self, gray_image, i , j, patchSize):
        """
            @args : gray_image --> Input grayscale image
            @args : i --> Row location of centre of patch being considered
            @args : j --> column location of centreo of patch being considered
            @args : patchSize: Size of patch 
            @retrun: List containing arrays of values of neighbhourhood patches computed based on shift coordinates
        """
        _NeighbourPatch = []
        for x in self.shifts:
            X = -(i + x[0]) + 1
            Y = -(j + x[1]) + 1
            b = np.roll(np.roll(gray_image,X, axis = 0),Y,axis=1)
            b = b[0:patchSize, 0:patchSize]
            _NeighbourPatch.append(b)
        return _NeighbourPatch

    def SSD(self, Wi, Wo):
        """
            @args Wi: neighbourhood window
            @args Wo: current window
            @return SSD: sum of squared differences of the matrix
        """
        x = Wi - Wo
        x = x ** 2
        return x.sum()

    def morvecCornerDetection(self, gray_image, patchSize):
        """
            @args: gray_image : input Gray_Scale Image
            @args: patchSize  : size of Patch
            @retrun Q : () Corners are detected from Q matrix
        """
        offset = patchSize // 2
        m, n = gray_image.shape 
        Q = np.zeros((m,n))
        for i in range(offset, m-offset):
            for j in range(offset, n-offset):
                Wo = gray_image[i-offset:i+offset+1, j-offset:j+offset+1]
                Wi = self.neighbourPatch(gray_image, i, j, patchSize)
                ssd = lambda window: self.SSD(Wo, window)
                ep = min(map(ssd, Wi))
                Q[i,j] = ep
        return Q

    def nonMaximumSuppression(self, Q):
        """
            @args : Q matrix --> from which Coreners points are deduced
            @return Q matrix --> NonMaximumSuppressed Q matrix (False Positives are removed)
        """
        for i in range(1,Q.shape[0]-1):
            for j in range(1, Q.shape[1]-1):
                patch = Q[i-1:i+2, j-1:j+2]
                if (np.max(patch) != Q[i,j]):
                    Q[i,j] = 0
        return Q

    def visualizeCornerPoints(self, Q, bgr_image):
        """
            @args : Q Matrix --> Coreners points are deduced from this matrix
            @args : bgr_image --> RGB IMAGE
            @return : bgr_image --> Color image marked with corner points
        """
        kps = np.argwhere(Q > 0.8875 * np.max(Q))
        print(0.8875 * np.max(Q))
        for p in kps:
            cv2.circle(bgr_image,(p[1],p[0]), color=(0, 0, 255), radius = 3)
        return bgr_image