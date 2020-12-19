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
img = cv2.imread("assignment4_2.jpg")
rgb_image = copy.deepcopy(img)
b,g,r = cv2.split(rgb_image)
bh = hist(b)
gh = hist(g)
rh = hist(r)
h,w,c = rgb_image.shape

# extraxt the coordinates and the correct pixel values from coords.txt and store them in a list
file = open("coords_2.txt","r")
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
    noise_b.append(b[row[i]][col[i]] - val[i][0]) 
    b[row[i]][col[i]] = val[i][0]
    noise_g.append(g[row[i]][col[i]] - val[i][1])
    g[row[i]][col[i]] = val[i][1]
    noise_r.append(r[row[i]][col[i]] - val[i][2])
    r[row[i]][col[i]] = val[i][2]
    i += 1

mean_b = np.mean(noise_b)
mean_g = np.mean(noise_g)
mean_r = np.mean(noise_r)

bH = hist(b)
gH = hist(g)
rH = hist(r)

output = cv2.merge((b,g,r))
cv2.imwrite('DenoisedImage_2.jpg',output)


fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [50,50]})
fig.set_figheight(4)
fig.set_figwidth(10)

axs[0].imshow(rgb_image)
axs[0].title.set_text('Given Noisy Image')

axs[1].imshow(output)
axs[1].title.set_text('Denoised Image')

fig.tight_layout()
plt.savefig('imageDenoising_2.jpg', dpi=300, bbox_inches='tight')
plt.show()

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
plt.savefig('HistofG&Dimages_2.jpg', dpi=300, bbox_inches='tight')
plt.show()



figure, ax = plt.subplots(1, 3)
figure.set_figheight(4)
figure.set_figwidth(10)
ax[0].plot(noise_b,color='b')
ax[0].title.set_text('Blue Channel')
ax[1].plot(noise_g,color='g')
ax[1].title.set_text('Green Channel')
ax[2].plot(noise_r,color='r')
ax[2].title.set_text('Red Channel')
figure.tight_layout()
figure.subplots_adjust(top=0.88)
figure.suptitle('Plot of Values Obtained from subtracting noisy pixel and original pixel values')
plt.savefig('Analysis_2.jpg', dpi=300)
plt.show()

print(mean_b,mean_g,mean_r)
var_b = np.var(noise_b)
var_g = np.var(noise_g)
var_r = np.var(noise_r)
print(var_b,var_g,var_r)

gauss_b = np.random.normal(mean_b,var_b ** 0.5,len(row))
gauss_g = np.random.normal(mean_g,var_g ** 0.5,len(row))
gauss_r = np.random.normal(mean_r,var_r ** 0.5,len(row))

p, c = plt.subplots(1, 3)
p.set_figheight(4)
p.set_figwidth(10)
c[0].plot(gauss_b,color='b')
c[0].title.set_text('Blue Channel')
c[1].plot(gauss_g,color='g')
c[1].title.set_text('Green Channel')
c[2].plot(gauss_r,color='r')
c[2].title.set_text('Red Channel')
p.tight_layout()
p.subplots_adjust(top=0.88)
p.suptitle('Plot of Values Obtained by generating gaussian noise with obtained mean and variance')
plt.savefig('proof_2.jpg', dpi=300)
plt.show()


median = cv2.medianBlur(rgb_image,3)
cv2.imshow('g',median)
cv2.waitKey(0)
hb = hist(median[:,:,0])
hg = hist(median[:,:,1])
hr = hist(median[:,:,2])
plt.plot(hb)
plt.plot(hg)
plt.plot(hr)
plt.show()







