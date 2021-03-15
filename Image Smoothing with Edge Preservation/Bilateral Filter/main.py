from bf_image_smoothing import Util, BilateralFilter

def main():
    util = Util()
    bf = BilateralFilter()

    fileName = "Input/assignment_6.jpg"

    filter_size = 5
    sigma_i = 50
    sigma_s = 50

    # Read the original Image
    org_image = util.imageRead(fileName)

    # Split the original Image into 3 channels
    b,g,r = util.splitBGR(org_image)

    # Pad the original Image based on the kernel size
    padded_image = util.padImage(org_image,filter_size)

    # Split the Padded Image into 3 channels
    pb,pg,pr = util.splitBGR(padded_image)

    # Call own_implemenatation function to smooth out B,G,R channels of images seperately 
    smoothB = bf.own_implementation(b,pb,filter_size,sigma_i,sigma_s)
    smoothG = bf.own_implementation(g,pg,filter_size,sigma_i,sigma_s)
    smoothR = bf.own_implementation(r,pr,filter_size,sigma_i,sigma_s)

    # merge the 3 smoothened channels to get single RGB image output
    output = util.mergeBGR(smoothB,smoothG,smoothR)
    
    # Display the smoothened output
    util.imageDisplay("Bilateral Filter", output)

    # store the output 
    util.imageWrite("Output/output.jpg",output)

    # opencv implemenatation of Bilateral Filter
    output_cv = bf.opencv_implementation(org_image, filter_size,sigma_i, sigma_s)
    util.imageDisplay("Bilateral Filter OpenCV", output_cv)
    util.imageWrite("Output/output_cv.jpg",output_cv)

if __name__ == '__main__':
    main()
