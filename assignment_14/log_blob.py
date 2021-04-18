# Import Neccesary Packages
import numpy as np 
import cv2
from matplotlib import pyplot as plt
import copy
import math
from scipy import signal

class Util:


    """
        Responsibilty : Read, Write, Display
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

class LOGBlobDetection:

    def scales(self, minimum_sigma,const_val, maximum_sigma = 20):
        i = 0
        scale = 0
        size_list = []
        scale_list = []
        while scale < maximum_sigma:
            scale = minimum_sigma * (const_val ** i) 
            if scale < maximum_sigma:
                scale_list.append(scale)
            i+=1
        for val in scale_list:
            x = np.round(val * 6)
            if x%2 == 0:
                x+=1
            size_list.append(int(x))
        
        return scale_list

    def LOG_generation(self, s):
        size = 2 * (np.floor(3 * s)) + 1
        x, y = np.mgrid[-size//2 + 1 : size // 2 + 1, -size//2 + 1 : size // 2 + 1]
        log = (-1/(2 * math.pi * (s ** 6)))*((x ** 2) + (y ** 2) - (s ** 2))* np.exp(-((x**2)+(y**2))/(2 * (s ** 2)))
        log = s * log 
        return log
    def ScaleNormalizedLOG(self, scales):
        SLOG = []
        for k in range(len(scales)):
            log = self.LOG_generation(scales[k])
            SLOG.append(log) 
        return SLOG

    def featureMap(self,logfilters, gray_image,t):
        response = []
        for k in range(len(logfilters)):
            r = signal.convolve2d(gray_image, logfilters[k], boundary='symm', mode='same')
            r = r ** 2
            r[r<t]=0
            response.append(r)
        return response

    def scaleSpatialNMS(self, featuremap, scales, h, w):
        kps = []
        for k in range(len(scales) - 2, 0,-1):
            for i in range(1,h-1):
                for j in range(1,w-1):
                    patchcurr = featuremap[k][i-1:i+2,j-1:j+2]
                    maxcurr = np.max(patchcurr)
                    if maxcurr == featuremap[k][i,j]:
                        maxminus = np.max(featuremap[k-1][i-1:i+2,j-1:j+2])
                        maxplus = np.max(featuremap[k+1][i-1:i+2,j-1:j+2])
                        if ((maxcurr >= maxminus) and (maxcurr >= maxplus)):
                            rad = int(np.sqrt(2) * scales[k])
                            kps.append((j,i,rad))
                        else:
                            featuremap[k][i,j] = 0
                    else:
                        featuremap[k][i,j] = 0 
        return kps 





        
