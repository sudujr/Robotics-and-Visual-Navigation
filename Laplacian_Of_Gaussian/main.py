from LoG import Util, LOGEdgeDetection
import numpy as np 

def main():
    util = Util()
    ed = LOGEdgeDetection()
    gray_image = util.imageRead("Input/assignment_9.jpg")
    #util.imageDisplay("Input 1",gray_image)
    bf_image = ed.imageBlur(gray_image, 11, 100, 100)
    #util.imageDisplay("Bilateral FIltered Image ",bf_image)
    pad_image = util.padImage(bf_image,5)

    #LoG Operator
    log = np.array([[0, 0, 1, 0, 0],[0, 1, 2, 1, 0],[1, 2, -16, 2, 1],[0, 1, 2, 1, 0],[0, 0, 1, 0, 0]])

    # 2nd Order Derivative
    grad = ed.imageConvolve(pad_image, log)

    # computing the magnitude of gradients
    out = ed.imageGradientMagnitude(grad)

    # output before zerocrosiing detection
    util.imageDisplay("Output Before Zero Crossing Detection",out)
    util.imageWrite("Output/output_before_zero_crossing.jpg",out)
    
    Zero_Out = ed.zeroCrossing(grad)
    util.imageDisplay("Output After Zero Crosiing Detection", Zero_Out)
    util.imageWrite("Output/output_after_zero_crossing.jpg",Zero_Out)


if __name__ == "__main__":
    main()
