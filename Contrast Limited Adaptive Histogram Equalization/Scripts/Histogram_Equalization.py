# Import Neccesary Packages
import numpy as np 
import cv2
from matplotlib import pyplot as plt
import copy

# Function to Compute Normalized Histogram
def nHist(image):
    m, n = image.shape
    h = [0] * 256
    for i in range(m):
        for j in range(n):
            h[image[i,j]]+= 1

    return np.asarray(h) * 1.0 / (m * n)

# Function to compute Cumulative Distribution Function
def cdfunction(hist):
    cdfi = 0
    c = [0.0] * 256
    for i in range(len(hist)):
        cdfi+= hist[i]
        c[i] = cdfi 
    return np.array(c)

# Function for Histogram Equalization
# return histogram equalized image, histogram before and after equalization
def histEqualization(image):
    nh = nHist(image) #normalized Histogram
    cdf = cdfunction(nh) #cdf
    T = np.uint8(255 * cdf) # Transfer function
    eImage = np.zeros_like(image)
    m, n = image.shape
    h = nh * (m * n)
    for i in range(m):
        for j in range(n):
            eImage[i,j] = T[image[i,j]]
    H = nHist(eImage) * (m * n)
    return eImage , h, H


# Load the Image and Store it in variable givenImage
givenImage = cv2.imread("/home/sudharshan/Documents/Robotics-and-Visual-Navigation/Contrast Limited Adaptive Histogram Equalization/Input/NightVision.jpg")

#cv2.imshow("Image", givenImage)
#cv2.waitKey(0)

# split the rgb image into seperate r, g, b images to equalise seperatelely
b, g, r = cv2.split(givenImage)

equalizedB , bh, bH = histEqualization(b)
equalizedG , gh, gH = histEqualization(g)
equalizedR , rh, rH = histEqualization(r)

# merge the seperately equlaized image to get the final Image
equalizedImage = cv2.merge((equalizedB,equalizedG,equalizedR))

# Display images and their Histogram
fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [50,50]})
fig.set_figheight(4)
fig.set_figwidth(10)

axs[0].imshow(givenImage)
axs[0].title.set_text('Given Image')
axs[0].axes.get_yaxis().set_visible(False)
axs[0].axes.get_xaxis().set_visible(False)
axs[1].imshow(equalizedImage)
axs[1].title.set_text('Histogram Equalized Image')
axs[1].axes.get_yaxis().set_visible(False)
axs[1].axes.get_xaxis().set_visible(False)
fig.tight_layout()
plt.savefig('/home/sudharshan/Documents/Robotics-and-Visual-Navigation/Contrast Limited Adaptive Histogram Equalization/InputandOutput/N-AHE/NightVision_NAHE_Images.jpg', dpi=300, bbox_inches='tight')
plt.show()


f, a = plt.subplots(1, 2, gridspec_kw={'width_ratios': [50,50]})
f.set_figheight(4)
f.set_figwidth(10)

a[0].plot(bh,color='b')
a[0].plot(gh,color='g')
a[0].plot(rh,color='r')
a[0].title.set_text('Given Image Histogram')
a[1].plot(bH,color='b')
a[1].plot(gH,color='g')
a[1].plot(rH,color='r')
a[1].title.set_text('Histogram Equalized Image Histogram')

fig.tight_layout()
plt.savefig('/home/sudharshan/Documents/Robotics-and-Visual-Navigation/Contrast Limited Adaptive Histogram Equalization/Histograms/N-AHE/NightVision_NAHE_HIST.jpg', dpi=300, bbox_inches='tight')

plt.show()






