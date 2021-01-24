# Import Neccesary Packages
import numpy as np 
import cv2
from matplotlib import pyplot as plt
import copy

class Util:


    """
        Responsibilty : Read, Write, Display, Convert to Binary, Expand
        Functions:
            imageRead() --> Reads the image from the system and gives us an grayscale image to perform operation
            imageWrite() --> Stores the image in the system
            imageDisplay() --> Displays the image to the user
            toBinary() --> Converts the given grayscale image into binary using Otsu’s Binarization
            padImage() --> Increases the size of Image to perform operation
    """
    
    def imageRead(self,fileName):
        """
            @args : fileName (String)
            @return : grayscale image (1 channel 2d array)
        """
        gray_image = cv2.imread(fileName, 0)
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

    def toBinary(self,pad_image):
        """
            @args: Grayscale Image
            @return : Binary Image

        """
        blur = cv2.GaussianBlur(pad_image,(5,5),0)
        ret3,binary_image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return binary_image

    def padImage(self,org_image,filter_size):
        """
            @args: original image and size of the filter to be operated on the original image
            @return : Expanded image
        """
        pad_size = int(filter_size/2)
        return cv2.copyMakeBorder( org_image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)



class Morph:
    """
        Responsibility : To perform Morphological Dilation and Erosion
        Functions:
            dilation() --> Performs Morphological dilation on the given Image
            erosion() --> Performs Morphological Erosion on the given Image
    """

    def dilation(self,binary_image,filter_size,org_image):
        """
            @args: binaryImage, filter size and OrginalImage
            @return : Dilated Image of Size original Image
        """
        R = np.zeros_like(org_image)
        size = (filter_size,filter_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)
        h,w = org_image.shape
        for i in range(h):
            for j in range(w):
                if np.logical_and(kernel,binary_image[i:i+filter_size,j:j+filter_size]).any():
                    R[i,j] = 255
        return R

    def erosion(self,binary_image,filter_size,org_image):
        """
            @args: binaryImage, filter size and OrginalImage
            @return : Eroded Image of Size original Image
        """
        R = np.zeros_like(org_image)
        size = (filter_size,filter_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)
        h,w = org_image.shape
        for i in range(h):
            for j in range(w):
                if np.array_equal(kernel, np.logical_and(kernel,binary_image[i:i+filter_size,j:j+filter_size])):
                    R[i,j] = 255
        return R


class OpenCVImplenetation:
    """
        open cv implementation of the above
    """
    def cv_dilation(self,image,filter_size):
        size = (filter_size,filter_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)
        return cv2.dilate(image,kernel,iterations = 1)
    
    def cv_erosion(self,image,filter_size):
        size = (filter_size,filter_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)
        return cv2.erode(image,kernel,iterations = 1)

    def cv_opening(self,image,filter_size):
        size = (filter_size,filter_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def cv_closing(self,image,filter_size):
        size = (filter_size,filter_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def cv_smoothing(self,image,filter_size):
        size = (filter_size,filter_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
