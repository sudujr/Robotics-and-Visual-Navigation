from moravec import Util, MoravecCornerDetection
import numpy as np 
import cv2
import copy

def main():
    util = Util()
    mcd = MoravecCornerDetection()

    # load the image  from the system as both gray scale image and rgb image
    bgr_image = util.imageReadBGR('Input/assignment_11.jpg')
    gray_image = util.imageReadGray('Input/assignment_11.jpg')
    # perform Moravec Corner Detection
    Q = mcd.morvecCornerDetection(gray_image, 5)
    Q = mcd.nonMaximumSuppression(Q)

    # Visualize and store the image marked with corner points
    img = mcd.visualizeCornerPoints(Q,bgr_image)
    util.imageDisplay(img, "OUTPUT MORAVEC")
    util.imageWrite(img,"Output/output_moravec.jpg")
    

if __name__ == '__main__':
    main()