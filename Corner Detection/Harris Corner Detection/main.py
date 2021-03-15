from harris import Util, HarrisCornerDetection
import numpy as np 
import cv2
import copy

def main():
    util = Util()
    hcd = HarrisCornerDetection()
    # SCRATCH IMPLEMENTATION
    # Load the Image as Gray Scale Image and Store the result in gray_image and Color image in bgr_image
    bgr_image = util.imageReadBGR('Input/assignment_12.jpg')
    bgr_image_copy = copy.deepcopy(bgr_image)
    gray_image = util.imageReadGray('Input/assignment_12.jpg')

    # Compute the Dimensions of gray scale Image
    m,n = gray_image.shape
    # compute harris corner response score for each neighbourhood (Thresholding is also done)
    M = hcd.harrisResponseScoreMatrix(gray_image)
    # Perform Non Maximum Suppression to remove false positives
    M = hcd.nonMaximumSuppression(M, m, n)

    # visualize the Corner points in the image
    bgr_image = hcd.visualizeCornerPoints(M,bgr_image)
    util.imageDisplay(bgr_image,"OUTPUT HARRIS")            
    #Save the imag in the system
    util.imageWrite(bgr_image, "Output/output_Harris.jpg")
    # OPENCV IMPLEMENTATION
    # refer https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
    gray = np.float32(gray_image)
    opencv_M = cv2.cornerHarris(gray,2,3,0.06)
    img = hcd.visualizeCornerPoints(opencv_M,bgr_image_copy,0)
    # visualize the Corner points in the image
    util.imageDisplay(img,"OUTPUT HARRIS CV")            
    #Save the imag in the system
    util.imageWrite(img, "Output/output_Harris_cv.jpg")

if __name__ == '__main__':
    main()