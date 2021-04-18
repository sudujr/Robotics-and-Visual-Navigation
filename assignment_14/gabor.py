import numpy as np 
import cv2 
from matplotlib import pyplot as plt 

Kernerl = cv2.getGaborKernel((50,50),8, (1/np.pi), 5, 0.5, 0 , ktype = cv2.CV_32F)
print(Kernerl)
plt.imshow(Kernerl)
plt.show()