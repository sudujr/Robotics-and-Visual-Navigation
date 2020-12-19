# Import Neccesary Packages
import numpy as np 
import cv2
from matplotlib import pyplot as plt
import copy

def nHist(image,a,b): 

    """

    Function to compte normalized hostogram of the given image
    accepts image and size of image as input
    return an numpy array of size L containing probability of each pixel occuring in this image

    """

    hist = [0] * 256  
    for i in range(a):
        for j in range(b):
            hist[image[i,j]]+= 1
    return np.asarray(hist) 

givenImage = cv2.imread('assignment3.jpg')
# Split the image into b,g,r
b,g,r = cv2.split(givenImage)
h,w = b.shape
bh = h*w*nHist(b,h,w)
gh = h*w*nHist(g,h,w)
rh = h*w*nHist(r,h,w)

clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(4,4))#increased values may cause noise
new_b = clahe.apply(b)
new_g = clahe.apply(g)
new_r = clahe.apply(r)

bH = h*w*nHist(new_b,h,w)
gH = h*w*nHist(new_g,h,w)
rH = h*w*nHist(new_r,h,w)


equalizedImage = cv2.merge((new_b,new_g,new_r))

# Display images and their Histogram
fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [50,50]})
fig.set_figheight(4)
fig.set_figwidth(10)

axs[0].imshow(givenImage)
axs[0].title.set_text('Given Image')

axs[1].imshow(equalizedImage)
axs[1].title.set_text('CLAHE Image')

fig.tight_layout()
plt.savefig('assignment_clahe_images.jpg', dpi=300, bbox_inches='tight')
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
a[1].title.set_text('CLAHE Image Histogram')
fig.tight_layout()
plt.savefig('assignment_clahe_hist.jpg', dpi=300, bbox_inches='tight')

plt.show()



