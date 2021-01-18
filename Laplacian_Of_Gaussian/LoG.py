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
            imageRead() --> Reads the image from the system and gives us an gray scale image to perform operation
            imageWrite() --> Stores the image in the system
            imageDisplay() --> Displays the image to the user
            padImage() --> Increases the size of Image to perform operation
    """
    
    def imageRead(self,fileName):
        """
            @args : fileName (String)
            @return : grayscale image (1 channel 2d array)
        """
        gray_image = cv2.imread(fileName,0)
        if gray_image is not None:
            return gray_image
    
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

    def padImage(self,org_image,filter_size):
        """
            @args: original image and size of the filter to be operated on the original image
            @return : Expanded image
        """
        pad_size = int(filter_size/2)
        return cv2.copyMakeBorder( org_image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT)

class LOGEdgeDetection:

    """
        Responsibilty : Perform Laplacian of Gaussian based edge detection
        Functions:
            imageBlur() --> Smoothens the image
            imageConvolve() --> Convolves the given image with the Filter 
            imageGradientMagnitude() --> Computes magnitude of gradient 
            zeroCrossing() --> Identifies edge points from the gradient based on human set threshold
    """
    
    def imageBlur(self,image, filterSize, sigma_i, sigma_s):

        """
            @args:
                image --> Graysacle Image
                filterSize --> Size of Kernel
                sigma_i --> Standard deviation for Intensity 
                sigma_s --> standard deviation for spatial Distance
            @return : filtered Image of size given image
        """
        return cv2.bilateralFilter(image, filterSize, sigma_i, sigma_s)

    def imageConvolve(self,pad_image,Kernel):

        """
            @args:
                pad_image --> padded image to accomodate sliding window of kernel for boundary cases
                kernel --> Filter

            @return : Filtered image of size orginal Image (traced from padded image)
        """
        imageH, imageW = pad_image.shape
        kernelH, kernelW = Kernel.shape
        P = int(kernelH/2)
        Q = int(kernelW/2)
        conv_out = np.zeros((imageH-(2*P), imageW-(2*Q)))
        for i in range(P, imageH-P):
            for j in range(Q, imageW-Q):
                val = 0
                for p in np.arange(-P, P+1):
                    for q in np.arange(-Q, Q+1):
                        val += pad_image[i+p, j+q] * Kernel[P-p, Q-q]
                conv_out[i-P,j-Q] = val
        return conv_out

    def imageGradientMagnitude(self, grad):
        """
            @args: GradX --> X Gradient of image, GradY --> Y Gradient of the Image
            @return : magnitude of the gradients
        """
        org_mag = np.abs(grad)
        org_mag /= np.max(org_mag)
        return np.uint8(255 * org_mag)

    def zeroCrossing(self, G):
        """
            @args: image: Gradient magnitude of Given Image operated on First order derivative Filter
            @redtrun: image with edge points highlighted
        """
        thres = np.absolute(G).mean() * 0.8 # referrd from https://stackoverflow.com/questions/22050199/python-implementation-of-the-laplacian-of-gaussian-edge-detection
        h,w = G.shape
        O = np.zeros((h,w))
        for i in range(1, h-1):
            for j in range(1, w-1):
                X = G[i-1:i+2,j-1:j+2]
                Minimum = -1 * np.min(X)
                Maximum = np.max(X)
                X[1,1] = 0
                X = X.flatten()
                neg = X[X<0]
                neg_count = neg.shape[0]
                pos = X[X>0]
                pos_count = pos.shape[0]
                if(neg_count > 0 and pos_count > 0):
                    if (Maximum - Minimum > thres):
                        O[i,j] = 255 
        return O
