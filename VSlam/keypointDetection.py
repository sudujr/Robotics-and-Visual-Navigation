#!/usr/bin/env python3

import cv2 
import numpy as np
from scipy import ndimage

image = cv2.imread("Input/test.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

class HarrisKeyPointDetection:

    def __init__(self):
        self.Sx = (1/8) * np.asarray([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        self.Sy = (1/8) * np.asarray([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])

    def jacobianComputation(self, image):
        Gx = ndimage.convolve(image, self.Sx, mode='constant', cval=0.0)
        Gy = ndimage.convolve(image, self.Sy, mode='constant', cval=0.0)
        self.Jxx = Gx ** 2
        self.Jxy = Gx * Gy 
        self.Jyy = Gy ** 2

    def StructureMatrix(self, neighbourSize):
        half = neighbourSize // 2

if __name__ == "__main__":
    image = cv2.imread("Input/test.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hcd = HarrisKeyPointDetection()
    hcd.gradientComputation(gray)
