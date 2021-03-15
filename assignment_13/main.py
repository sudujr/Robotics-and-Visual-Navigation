from shi_tomasi import Util, ShiTomasiCornerDetection
import numpy as np 
import cv2
import copy

def main():
    util = Util()
    scd = ShiTomasiCornerDetection()
    # SCRATCH IMPLEMENTATION
    # Load the Image as Gray Scale Image and Store the result in gray_image and Color image in bgr_image
    bgr_image = util.imageReadBGR('assignment_13.jpg')
    bgr_image_copy = copy.deepcopy(bgr_image)
    gray_image = util.imageReadGray('assignment_13.jpg')

    # Compute the Dimensions of gray scale Image
    m,n = gray_image.shape
    # compute harris corner response score for each neighbourhood (Thresholding is also done)
    M = scd.shiResponseScoreMatrix(gray_image)
    # Perform Non Maximum Suppression to remove false positives
    M = scd.nonMaximumSuppression(M, m, n)
    kps = scd.goodFeatures(M)
    # visualize the Corner points in the image
    bgr_image = scd.visualizeCornerPoints(kps,bgr_image)
    util.imageDisplay(bgr_image,"OUTPUT SHI TOMASI")            
    #Save the imag in the system
    util.imageWrite(bgr_image, "output_Shi_Tomasi.jpg")
    # OPENCV IMPLEMENTATION
    # refer https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
    corners = cv2.goodFeaturesToTrack(gray_image,500,0.01,10)
    corners = np.int0(corners)
    for p in corners:
        x,y = p.ravel()
        cv2.circle(bgr_image_copy,(x,y), color=(0, 0, 255), radius = 3)
    # visualize the Corner points in the image
    util.imageDisplay(bgr_image_copy,"OUTPUT HARRIS CV")            
    #Save the imag in the system
    util.imageWrite(bgr_image_copy, "output_Shi_Tomasi_cv.jpg")

if __name__ == '__main__':
    main()