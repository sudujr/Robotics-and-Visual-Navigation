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
            imageStoreDisplay() --> Displays the output and stores the Reslut
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


    def padImage(self,org_image,filter_size):
        """
            @args: original image and size of the filter to be operated on the original image
            @return : Expanded image
        """
        pad_size = int(filter_size/2)
        return cv2.copyMakeBorder( org_image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT)

class CannyEdgeDetection:

    """
        Responsibility :  To perform canny edge edtection 
        Functions :
            imageBlur() --> to smooth the image
            imageConcolve() -- > to concolve an image with the given filter
            imageGradientMagnitude() --> to compute the magnitude of the gradients
            imageGradientOrientation() --> Direction of Gradient
            disctreOrientation() --> Disctring the direction of gradient to perform Non maximum Suppression
            nonMaxSuppression() --> Non maximal Suppression to remove False positive points
            doubleThresholding() --> To find strong edge points, weak edge points and False points
    """
    def imageBlur(self, image, filter_size, sigma):

        """
            @args:
                image --> Gray Scale image
                filter_size --> size of gaussian kernel
            @return : smoothened image with size same as image
        """
        return cv2.GaussianBlur(image,filter_size,sigma,cv2.BORDER_DEFAULT)    
    
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
        conv_out /= np.max(conv_out)
        return conv_out
    
    def imageGradientMagAndAngle(self, gradX, gradY):
        """
            @args: GradX --> X Gradient of image, GradY --> Y Gradient of the Image
            @return : magnitude and orientation of gradients of the gradients
        """
        org_mag = np.sqrt(np.power(gradX,2)+ np.power(gradY,2))
        org_mag /= np.max(org_mag)
        angle_out = np.zeros_like(org_mag)
        h,w = gradX.shape
        for i in range(h):
            for j in range(w):
                if gradX[i,j] == 0:
                    if gradY[i,j] < 0:
                        angle_out[i,j] = -90
                    else:
                        angle_out[i,j] = 90
                else:
                    angle_out[i,j] = np.arctan(gradY[i,j]/gradX[i,j])
        G = copy.deepcopy(angle_out)
        G[(G > -22.5) & (G <= 22.5)] = 0
        G[(G > 22.5) & (G <= 67.5)] = 45
        G[(G > 67.5) & (G <= 90)] = 90
        G[(G >= -90) & (G <= -67.5)] = 90
        G[(G > -67.5) & (G <= -22.5)] = 135
        return org_mag, angle_out, np.uint8(G)

    def scaleMagnitude(self, gradMag):
        """
            @args: magnitufde of Gradients
            @return : scaled output
        """
        return np.uint8(255* gradMag)

    def nonMaxSuppression(self,M, A):
        """
            @args: Magnitude of Gradient and Discretised Orientation of Gradient
            @return: Magnitude with removal of false positives
        """
        out = copy.deepcopy(M)
        h,w = M.shape
        for i in range(1,h-1):
            for j in range(1,w-1):
                if A[i,j] == 0:
                    if (M[i,j] < M[i,j-1] or M[i,j] < M[i,j+1]):
                        out[i,j] = 0
                elif A[i,j] == 45:
                    if (M[i,j] < M[i+1,j-1] or M[i,j] < M[i-1,j+1]):
                        out[i,j] = 0
                elif A[i,j] == 90:
                    if (M[i,j] < M[i+1,j] or M[i,j] < M[i-1,j]):
                        out[i,j] = 0
                elif A[i,j] == 135:
                    if (M[i,j] < M[i-1,j-1] or M[i,j] < M[i+1, j+1]):
                        out[i,j] = 0
        
        return out

    def doubleThresholding(self, NMSout):
        hratio = 0.15
        lratio = 0.1
        highThreshold = np.max(NMSout) * hratio
        lowThreshold = highThreshold * lratio
        NMSout[NMSout >= highThreshold] = 255
        NMSout[(NMSout > lowThreshold) & (NMSout < highThreshold)] = 128
        NMSout[(NMSout <= lowThreshold)] = 0
        return NMSout

    def hysterisis(self, DTOut):
        h,w = DTOut.shape
        for i in range(1,h-1):
            for j in range(1,w-1):
                if(DTOut[i,j] == 128):
                    if (DTOut[i-1,j-1] == 255 or DTOut[i-1,j] == 255 or DTOut[i-1,j+1] == 255 or DTOut[i, j-1] == 255 or DTOut[i, j+1] == 255 or DTOut[i+1,j-1] == 255 or DTOut[i+1,j] == 255 or DTOut[i+1, j+1] == 255):
                        DTOut[i,j] = 255
                    else:
                        DTOut[i,j] = 0
        return DTOut
