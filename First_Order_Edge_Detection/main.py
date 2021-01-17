from edge_detection import Util, FistOrderEdgeDetection
import numpy as np 

def main():
    util = Util()
    ed = FistOrderEdgeDetection()
    gray_image_1 = util.imageRead("Input/assignment_8_1.jpg")
    gray_image_2 = util.imageRead("Input/assignment_8_2.jpg")
    #util.imageDisplay("Input 1",gray_image_1)
    #util.imageDisplay("Input 2",gray_image_2)
    bf_image_1 = ed.imageBlur(gray_image_1, 5, 100, 25)
    bf_image_2 = ed.imageBlur(gray_image_2, 5, 100, 25)
    #util.imageDisplay("Bilateral FIltered Image 1",bf_image_1)
    #util.imageDisplay("Bilateral Filtered Image 2", bf_image_2)
    pad_image_1 = util.padImage(bf_image_1,3)
    pad_image_2 = util.padImage(bf_image_2,3)
    
    #Scharr Operator
    SCx = np.array([[-3, 0, 3],[-10, 0, 10],[-3, 0, 3]])
    SCy = np.array([[-3, -10, -3],[0, 0, 0],[3, 10, 3]])

    # Derivative along X and Y axis
    gradX1 = ed.imageConvolve(pad_image_1, SCx)
    gradY1 = ed.imageConvolve(pad_image_1, SCy)

    gradX2 = ed.imageConvolve(pad_image_2, SCx)
    gradY2 = ed.imageConvolve(pad_image_2, SCy)

    # computing the magnitude of gradients
    out1 = ed.imageGradientMagnitude(gradX1,gradY1)
    out2 = ed.imageGradientMagnitude(gradX2,gradY2)

    out1 = ed.edgeScaling(out1)
    out2 = ed.edgeScaling(out2)
    util.imageDisplay("Output",out1)
    util.imageDisplay("Output",out2)

    util.imageWrite("Output/Output1.jpg",out1)
    util.imageWrite("Output/Output2.jpg",out2)


if __name__ == "__main__":
    main()