# Import Neccesary Packages
import numpy as np 
import cv2
from matplotlib import pyplot as plt
import copy

#Load the given image
givenImage = cv2.imread('assignment3.jpg')

# Split the image into b,g,r
b,g,r = cv2.split(givenImage)


def imageCrop(image,a,b,n):

    """

    Function to split the image into n * n sub images
    Function accepts the Full Images, size of each subimage(a,b) and number of subimages(n * n)
    The divided subimages are stored in n*n List

    """ 

    G = []
    starti = 0
    for i in range(n):
        startj = 0
        g = []
        for j in range(n):
            g.append(image[starti:starti+a,startj:startj+b])
            startj+=b 
        G.append(g)
        starti+=a 
    return G 

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
    return np.asarray(hist) * 1.0 / (a * b)

def cdFunction(nhist):
    """

    Function to compute cumulative distribution function for given Image
    accepts normalized histogram of an image as input
    returns array of size L contatining cdf value
    
    """

    cdfi = 0
    c = [0.0] * 256
    for i in range(len(nhist)):
        cdfi+= nhist[i]
        c[i] = cdfi 
    return np.asarray(c)



def setTransferFunction(image,n,a,b):

    """

    Function to compute the set of Transferfunction of n*n divided subimages
    accepts the original image, size of each subimage and n as input
    calls image crop function to divide the images into subimages
    For each subimage, computes the Transfer Function and stores it in a n*n List
    return 2d list contatining transferfunction of each subimage

    """

    G = imageCrop(image,a,b,n)
    T = []
    for i in range(n):
        t = []
        for j in range(n):
            nhist = nHist(G[i][j],a,b)
            cdf = cdFunction(nhist)
            t.append(np.uint8( 255 * cdf))
        T.append(t)
    return T 

def ahe(image,n):

    """

    FUnction to perform adaptive Histogram Equalization
    Accepts image and n as Input
    return the equlaized image
    """

    h,w = image.shape 
    a = int(h/n)
    b = int(w/n)
    mida = int(a/2)
    midb = int(b/2)
    bega = mida 
    begb = midb
    enda = h - mida
    endb = w - midb
    vnew = np.zeros_like(image)
    T = setTransferFunction(image,n,a,b)
    for i in range(bega,enda):
        for j in range(begb,endb):
            li = int((i - mida) / a)
            lj = int((j - midb) / b)
            mi = li + 1
            mj = lj + 1
            y = (i - ((li * a) + mida)) / a 
            x = (j - ((lj * b) + midb)) / b 
            vnew[i,j] = int((y*x*T[mi][mj][image[i,j]]) + (y*(1-x)*T[mi][lj][image[i,j]]) + ((1-y)*x*T[li][mj][image[i,j]]) + ((1-y)*(1-x)*T[li][lj][image[i,j]]))
    for i in range(bega):
        for j in range(begb):
            vnew[i,j] = T[0][0][image[i,j]]       
    for i in range(bega):
        for j in range(begb,endb):
            li = int(i/a)
            lj = int((j - midb) / b)
            mj = lj + 1
            y = 2*i / a
            x = (j - ((lj * b) + midb)) / b 
            vnew[i,j] = int(x*T[li][mj][image[i,j]] + (1-x)*T[li][lj][image[i,j]])
    for i in range(bega,enda):
        for j in range(begb):
            li = int((i - mida) / a)
            lj = int(j/b)
            mi = li+1
            y = (i - ((li * a) + mida)) / a 
            x = 2*j/b 
            vnew[i,j] = int(y*T[mi][lj][image[i,j]] + (1-y)*T[li][lj][image[i,j]])
    for i in range(enda,h):
        for j in range(begb): 
            li = int(i/a)
            lj = int(j/b)
            y = 2*(i - ((li * a) + mida)) / a 
            x = 2*j/b         
            vnew[i,j] = int(T[li][lj][image[i,j]])
    for i in range(enda,h):
        for j in range(begb,endb):
            li = int(i/a)
            lj = int((j - midb) / b)
            mj = lj+1
            y = 2*(i - ((li * a) + mida)) / a 
            x = (j - ((lj * b) + midb)) / b 
            vnew[i,j] =int((x*T[li][mj][image[i,j]] + (1-x)*T[li][lj][image[i,j]]))
    for i in range(enda,h):
        for j in range(endb,w):
            li = int(i/a)
            lj = int(j/b)
            y = (i - ((li * a) + mida)) / a 
            x = (j - ((lj * b) + midb)) / b 
            vnew[i,j] = int(T[li][lj][image[i,j]])
    for i in range(bega,enda):
        for j in range(endb,w):
            li = int((i - mida) / a)
            lj = int(j/b)
            mi = li+1
            y = (i - ((li * a) + mida)) / a 
            x = 2*(j - ((lj * b) + midb)) / b 
            vnew[i,j] = int(y*T[mi][lj][image[i,j]] + (1-y)*T[li][lj][image[i,j]])

    for i in range(bega):
        for j in range(endb,w):
            li = int(i/a)
            lj = int(j/b)
            y = 2*i/a 
            x = 2*(j - ((lj * b) + midb)) / b 
            vnew[i,j] = int((T[li][lj][image[i,j]]))
    return vnew
n=4
h,w = b.shape
bh = h*w*nHist(b,h,w)
gh = h*w*nHist(g,h,w)
rh = h*w*nHist(r,h,w)

new_b = ahe(b,n)
new_g = ahe(g,n)
new_r = ahe(r,n)

bH = h*w*nHist(new_b,h,w)
gH = h*w*nHist(new_g,h,w)
rH = h*w*nHist(new_r,h,w)


equalizedImage = cv2.merge((new_b,new_g,new_r))
cv2.imshow('e',equalizedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Display images and their Histogram
fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [50,50]})
fig.set_figheight(4)
fig.set_figwidth(10)

axs[0].imshow(givenImage)
axs[0].title.set_text('Given Image')

axs[1].imshow(equalizedImage)
axs[1].title.set_text('Adaptive Equalized Image')

fig.tight_layout()
plt.savefig('assignment_ahe_images.jpg', dpi=300, bbox_inches='tight')
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
a[1].title.set_text('Adpative Histogram Equalized Image Histogram')
fig.tight_layout()
plt.savefig('assignment_ahe_hist.jpg', dpi=300, bbox_inches='tight')

plt.show()

