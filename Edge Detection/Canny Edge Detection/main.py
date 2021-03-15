from edge_detection import Util, CannyEdgeDetection
import numpy as np 
import cv2

def main():
    util = Util()
    ced = CannyEdgeDetection()

    # Load the Image as Gray Scal Image and Store the result in gray_image
    gray_image = util.imageRead('Input/assignment_10.jpg')

    # Step1 : Gaussian Blurring
    gauss_out = ced.imageBlur(gray_image, (5,5), 25)

    # Display the Gaussian Blurred Image
    util.imageDisplay(gauss_out,"Output_GaussainBlurring")
    util.imageWrite(gauss_out,"Output/Output_GaussainBlurring.jpg")

    # pad the blurred image to accomodate Convolutaion in border cases
    padd_out = util.padImage(gauss_out,3)

    # create Sobel filter in x and y direction
    Sx = (1/8) * np.asarray([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    Sy = (1/8) * np.asarray([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])

    # step 2a: Perform Convolution of Sx and Sy with padded image to get gradients in X and Y direction
    gradX = ced.imageConvolve(padd_out, Sx)
    gradY = ced.imageConvolve(padd_out, Sy)

    gradX_disp = ced.scaleMagnitude(gradX)
    gradY_disp = ced.scaleMagnitude(gradY)

    # Display the results
    util.imageDisplay(gradX_disp,"Output_GradientX")
    util.imageWrite(gradX_disp,"Output/Output_GradientX.jpg")
    util.imageDisplay(gradY_disp,"Output_GradientY")
    util.imageWrite(gradY_disp,"Output/Output_GradientY.jpg")


    # step2b Compute the magnitude and discretized orientation of the gradients
    mag_grad, ang_grad, G = ced.imageGradientMagAndAngle(gradX, gradY)
    mag_grad = ced.scaleMagnitude(mag_grad)

    #Display the Magnitude of Gradients
    util.imageDisplay(mag_grad, "Output_GradientMagnitude")
    util.imageWrite(mag_grad, "Output/Output_GradientMagnitude.jpg")

    #step3: Non Maximum Suppresseion to remove false positive points
    
    nms_out = ced.nonMaxSuppression(mag_grad, G)

    # Display the output
    util.imageDisplay(nms_out, "Output_NonMaximumSuppression")
    util.imageWrite(nms_out, "Output/Output_NonMaximumSuppression.jpg")

    # step4 : Double thresholding
    dt_out = ced.doubleThresholding(nms_out)

    # Display the results
    util.imageDisplay(dt_out, "Output_DoubleThresholding")
    util.imageWrite(dt_out, "Output/Output_DoubleThresholding.jpg")


    # Step5 Edge tracking

    Canny_out = ced.hysterisis(dt_out)

    # Display the results
    util.imageDisplay(Canny_out, "Output_Hystersis")
    util.imageWrite(Canny_out, "Output/Output_Hystersis.jpg")


    X = cv2.Canny(gray_image,150,180)
    util.imageDisplay(X, "Output_OpenCV")
    util.imageWrite(X, "Output/Output_OpenCV.jpg")

    

    

if __name__ == '__main__':
    main()