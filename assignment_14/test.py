# Import Neccesary Packages
import numpy as np 
import cv2
from matplotlib import pyplot as plt
import copy
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

    def scales(self, minimum_sigma,const_val, maximum_sigma = 40):
        i = 0
        scale = 0
        scales = []
        while scale < maximum_sigma:
            scale = minimum_sigma * (const_val ** i) 
            if scale < maximum_sigma:
                scales.append(scale)
            i+=1
        sizes = [2 * (np.ceil(scale * 3)) + 1 for scale in scales]
        
        return scales, sizes

    def generateScaledLoGs(self, scales, sizes):
        SLOG = []
        for (scale, size) in zip(scales, sizes):
            X, Y = np.mgrid[-size//2 + 1 : size // 2 + 1, -size//2 + 1 : size // 2 + 1]
            slog = scale * (1/(2 * np.pi * (scale ** 6)))*((X ** 2) + (Y ** 2) - (scale ** 2))* np.exp(-((X**2)+(Y**2))/(2 * (scale ** 2)))
            SLOG.append(slog)
        return SLOG

    def responseMaps(self, gray_image, SLOG):
        featureMap = []
        for scaledlog in SLOG:
            response = signal.convolve2d(gray_image, scaledlog, boundary='symm', mode='same')
            featureMap.append(response)
        return np.asarray(featureMap)

    def getCandidateKeypoints(self, responseMatrix, threshold, scales):
        kps = []
        s, h, w = responseMatrix.shape
        for k in range(s-2,0,-1):
            for i in range(8, h-8):
                for j in range(8, w-8):
                    cur_val = responseMatrix[k,i,j]
                    if np.abs(cur_val) > threshold:
                        patch = responseMatrix[k-1:k+2, i-1:i+2, j-1:j+2]
                        if (np.argmax(patch) == 13 or np.argmin(patch)==13):
                            kps.append([int(j),int(i),int(np.sqrt(2)*scales[k])])
        return kps

def main():
    util = Util()
    logblob = LOGBlobDetection()
    rgb_image = util.imageReadBGR('HubbleDeepField.png')
    gray_image = util.imageReadGray('HubbleDeepField.png')
    h, w = gray_image.shape
    sigma0 = 2.67
    k = 1.212
    threshold = 0.1
    scales, sizes = logblob.scales(sigma0, k)    
    kernels = logblob.generateScaledLoGs(scales,sizes)
    """
    for idx,kernel in enumerate(kernels):
        plt.imshow(kernel)
        plt.axis('off')
        plt.savefig("PositiveLogFilter%d"%(sizes[idx]), bbox_inches='tight')
        plt.show()
    """

    
    a = logblob.responseMaps(gray_image,kernels)
    print(a.shape)
    for g in a:
        print(g.max())

    """
    for x in a:
        plt.imshow(x, cmap="gray")
        plt.show()
    """
    
    kps = logblob.getCandidateKeypoints(a, threshold, scales)
    for p in kps:
        cv2.circle(rgb_image,(p[0],p[1]),p[2],(0, 255, 255),1)
    util.imageDisplay(rgb_image, "OUTPUT")
    

if __name__ == '__main__':
    main()