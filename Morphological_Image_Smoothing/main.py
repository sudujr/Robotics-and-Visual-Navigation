from morphological_image_smoothing import Util, Morph, OpenCVImplenetation

def main():

    util = Util()
    morph = Morph()
    openCVImplenetation = OpenCVImplenetation()

    filename = 'Input/img_3327_o.jpg'

    org_image = util.imageRead(filename)
    kernel_size = 5
    pad_image = util.padImage(org_image,kernel_size)

    # Converting original grayscale Image to Binary Image
    org_binary_image = util.toBinary(org_image)

    # Converting Padded Grayscale Image to Binary Image
    pad_binary_image = util.toBinary(pad_image)
    util.imageDisplay("BinaryInput",org_binary_image)
    util.imageWrite("Output/Binary_Image.jpg",org_binary_image)

    # Morphological Dilation
    dilated_image = morph.dilation(pad_binary_image,kernel_size,org_image)
    util.imageDisplay("Morphological Dilation",dilated_image)
    util.imageWrite("Output/Dilation_S.jpg",dilated_image)

    # Morphological Erosion
    eroded_image = morph.erosion(pad_binary_image,kernel_size,org_image)
    util.imageDisplay("Morphological Erosion",eroded_image)
    util.imageWrite("Output/Erosion_S.jpg",eroded_image)

    # Morphological Opening
    o_erode_image = morph.erosion(pad_binary_image,kernel_size,org_image)
    o_pad_image = util.padImage(o_erode_image,kernel_size)
    morph_opening = morph.dilation(o_pad_image,kernel_size,o_erode_image)
    util.imageDisplay("Morphological Opening",morph_opening)
    util.imageWrite("Output/Opening_S.jpg",morph_opening)

    # Morphological Closing
    c_dilated_image = morph.dilation(pad_binary_image,kernel_size,org_image)
    c_pad_image = util.padImage(c_dilated_image,kernel_size)
    morph_closing = morph.erosion(c_pad_image,kernel_size,c_dilated_image)
    util.imageDisplay("Morphological Closing",morph_closing)
    util.imageWrite("Output/Closing_S.jpg",morph_closing)   

    # Morphological Image Smoothing
    io_erode_image = morph.erosion(pad_binary_image,kernel_size,org_image)
    io_pad_image = util.padImage(io_erode_image,kernel_size)
    io_dilate_image = morph.dilation(io_pad_image,kernel_size,io_erode_image)
    ic_pad_image = util.padImage(io_dilate_image,kernel_size)
    ic_dilate_image = morph.dilation(ic_pad_image,kernel_size,io_dilate_image)
    ic_pad_image = util.padImage(ic_dilate_image,kernel_size)
    morph_image_smoothing = morph.erosion(ic_pad_image,kernel_size,ic_dilate_image)
    util.imageDisplay("Morphological Image Smoothing",morph_image_smoothing)
    util.imageWrite("Output/Morph_S.jpg",morph_image_smoothing)

    # Opencv Implementation of the above same
    # DILATION

    dil_img = openCVImplenetation.cv_dilation(org_binary_image,kernel_size)
    util.imageDisplay("Morphological Dilation CV",dil_img)
    util.imageWrite("Output/Dilation_CV.jpg",dil_img)

    # EROSION

    erd_img = openCVImplenetation.cv_erosion(org_binary_image,kernel_size)
    util.imageDisplay("Morphological Erosion CV", erd_img)
    util.imageWrite("Output/Erosion_CV.jpg",erd_img)

    # OPENING

    opening = openCVImplenetation.cv_opening(org_binary_image,kernel_size)
    util.imageDisplay("Morphological Opening CV",opening)
    util.imageWrite("Output/Opening_CV.jpg",opening)

    # CLOSING

    closing = openCVImplenetation.cv_closing(org_binary_image,kernel_size)
    util.imageDisplay("Morphological Closing CV",closing)
    util.imageWrite("Output/Closing_CV.jpg",closing)   

    # IMAGE SMOOTHING

    morph_smoothing = openCVImplenetation.cv_smoothing(org_binary_image,kernel_size)
    util.imageDisplay("Morphological Image Smoothing CV",morph_smoothing)
    util.imageWrite("Output/Morph_CV.jpg",morph_smoothing)

    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    print(kernel)
    """


if __name__ == '__main__':
    main()