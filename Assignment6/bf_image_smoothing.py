# Import Neccesary Packages
import numpy as np 
import cv2
from matplotlib import pyplot as plt
import copy
import math


def imagePadding(image,pad_size,mode):
    return np.pad(image,((pad_size,pad_size),(pad_size,pad_size),(0,0)),mode)

def l2Norm(ni,nj,ci,cj):
    return np.sqrt(((ci-ni)**2) + ((cj-nj)**2))

def gaussian(value,sigma):
    constant = (1.0 / (2 * np.pi * (sigma ** 2)) ** (1/2))
    return  constant * math.exp( (-1/2)*(value/sigma) ** 2)

def filter_processing(image,padded_image,bf_image,loci,locj,filter_size,pad_size,sigma_space,sigma_colour):
    w = 0
    filter_value = 0
    for i in range(filter_size):
        for j in range(filter_size):
            x = loci + i 
            y = loci + j 
            li = (i - pad_size) + loci 
            lj = (j - pad_size) + loci
            g_color = gaussian((int(image[loci][locj]) - int(padded_image[x][y])),sigma_colour)
            g_space = gaussian(l2Norm(li,lj,loci,locj), sigma_space)
            w_part = g_color * g_space
            filter_value += padded_image[x][y] * w_part
            w+= w_part
    filter_value/= w 
    bf_image[loci,locj] = int(filter_value)
    return bf_image

def bilateral_filter(image,padded_image,filter_size,sigma_space,sigma_colour):
    bf_image = np.zeros_like(image)
    h,w = image.shape
    pad_size = int(filter_size / 2)
    print(pad_size)
    for i in range(h):
        for j in range(w):
            bf_image = filter_processing(image, padded_image,bf_image,i,j,filter_size,pad_size,sigma_space,sigma_colour)
    return bf_image

image = cv2.imread("assignment_6.jpg")
padded_image = imagePadding(image,3,mode = 'edge')
b,g,r = cv2.split(image)
bp,gp,rp = cv2.split(padded_image)

bf = bilateral_filter(b,bp,7,100,100)
gf = bilateral_filter(g,gp,7,100,100)
rf = bilateral_filter(r,rp,7,100,100)

output = cv2.merge((bf,gf,rf))
cv2.imshow('output',output)
cv2.waitKey(0)
cv2.imwrite('Output.jpg',output)



    

