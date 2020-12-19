# Import Neccesary Packages
import numpy as np 
import cv2
from matplotlib import pyplot as plt
import copy

# Function to Compute Normalized Histogram
def hist(image):
    m, n = image.shape
    h = [0] * 256
    for i in range(m):
        for j in range(n):
            h[image[i,j]]+= 1

    return np.asarray(h) 

# Load the Image Using cv2.imread
img = cv2.imread("assignment4_1.jpg")
rgb_image = copy.deepcopy(img)
b,g,r = cv2.split(rgb_image)
bh = hist(b)
gh = hist(g)
rh = hist(r)
h,w,c = rgb_image.shape


# extraxt the coordinates and the correct pixel values from coords.txt and store them in a list
file = open("coords_1.txt","r")
c = 0
col = []
row = []
val = []
for f in file:
    x = f.split(",")
    row.append(int(x[0][1:]))
    col.append(int(x[1][:-1].strip()))
    z = ' '.join(x[2].split())
    z = z[1:-1].strip()
    z = z.split(" ")
    z =[int(i) for i in z]
    val.append(z)
    c += 1
file.close()

i=0


noise_b = []
noise_g = []
noise_r = []

while i < len(row):
    noise_b.append(b[row[i]][col[i]])
    b[row[i]][col[i]] = val[i][0]
    noise_g.append(g[row[i]][col[i]])
    g[row[i]][col[i]] = val[i][1]
    noise_r.append(r[row[i]][col[i]])
    r[row[i]][col[i]] = val[i][2]
    i += 1

mean_b = np.mean(noise_b)
mean_g = np.mean(noise_g)
mean_r = np.mean(noise_r)

output = cv2.merge((b,g,r))


bH = hist(b)
gH = hist(g)
rH = hist(r)

"""
f, a = plt.subplots(1, 2, gridspec_kw={'width_ratios': [50,50]})
f.set_figheight(4)
f.set_figwidth(10)

a[0].plot(bh,color='b')
a[0].plot(gh,color='g')
a[0].plot(rh,color='r')
a[0].title.set_text('Given Noisy Image Histogram')

a[1].plot(bH,color='b')
a[1].plot(gH,color='g')
a[1].plot(rH,color='r')
a[1].title.set_text('Denoised Image Histogram')
f.tight_layout()
plt.savefig('assignment_he_hist.jpg', dpi=300, bbox_inches='tight')

plt.show()



# Display images and their Histogram
fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [50,50]})
fig.set_figheight(4)
fig.set_figwidth(10)

axs[0].imshow(rgb_image)
axs[0].title.set_text('Given Noisy Image')

axs[1].imshow(output)
axs[1].title.set_text('Denoised Image')

fig.tight_layout()
plt.savefig('noiseRemoval.jpg', dpi=300, bbox_inches='tight')
plt.show()
"""
median = cv2.medianBlur(output,3)
mbH = hist(median[:,:,0])
mgH = hist(median[:,:,1])
mrH = hist(median[:,:,2])


f, a = plt.subplots(1, 2, gridspec_kw={'width_ratios': [50,50]})
f.set_figheight(4)
f.set_figwidth(10)

a[0].imshow(median)
a[0].title.set_text('3 X 3 Median Filtered Image')

a[1].plot(mbH,color='b')
a[1].plot(mgH,color='g')
a[1].plot(mrH,color='r')
a[1].title.set_text('Histogram of 3 X 3 Median Filtered Image')
f.tight_layout()
plt.savefig('Dmedian3*3.jpg', dpi=300, bbox_inches='tight')

plt.show()

cv2.imshow('median',median)
cv2.waitKey(0)
