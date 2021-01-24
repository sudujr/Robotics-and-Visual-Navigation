from kw_image_smoothing import Util, KuwaharaFilter

def main():
    util = Util()
    kw = KuwaharaFilter()

    filter_size = 11

    fileName = "Input/assignment_7.jpg"
    org_image = util.imageRead(fileName)
    hsv_image = util.toHSV(org_image)
    h,s,v = util.splitHSV(hsv_image)
    pad_v = util.padImage(v,filter_size)
    new_v = kw.implementation(v,pad_v,filter_size)
    new_hsv = util.mergeHSV(h,s,new_v)
    output = util.toBGR(new_hsv)
    util.imageDisplay("Kuwahara Filter Output",output)
    util.imageWrite("Output/output.jpg",output)


if __name__ == '__main__':
    main()