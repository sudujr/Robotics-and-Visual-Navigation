from log_blob import Util, LOGBlobDetection
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import cv2 

def main():
    util = Util()
    logblob = LOGBlobDetection()
    rgb_image = util.imageReadBGR('assignment_14.jpg')
    gray_image = util.imageReadGray('assignment_14.jpg')
    h, w = gray_image.shape
    sigma0 = 1.6
    k = 1.414
    threshold = 100
    scales = logblob.scales(sigma0, k)
    print(scales)
    
    r = logblob.ScaleNormalizedLOG(scales)
    for i in range(len(r)):
        print(np.max(r[i]))
        print(r[i].shape)
        plt.imshow(r[i])
        plt.show()
    a = logblob.featureMap(r,gray_image,threshold)
    for x in range(len(r)):
        print(np.max(a[x]))
        print(a[x].shape)
        plt.imshow(a[x])
        plt.show()
    kps = logblob.scaleSpatialNMS(a, scales, h, w)
    for p in kps:
        cv2.circle(rgb_image,(p[0],p[1]),p[2],(0, 0, 255),1)
    util.imageDisplay(rgb_image, "OUTPUT")


if __name__ == '__main__':
    main()