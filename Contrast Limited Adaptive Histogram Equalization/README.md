<p align="center"><img width=40% src="https://github.com/sudujr/Robotics-and-Visual-Navigation/blob/main/Contrast%20Limited%20Adaptive%20Histogram%20Equalization/Media/a9bc9c907620e8a74f6239a33d47bf90.jpg"></p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# CONTRAST ENHANCEMENT USING HISTOGRAM 
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
## Overview 

Images are captured by differnt types of camera's under differnent physical and atmospheric conditions. If the Image is shot under low light and less contrast conditions it will not be able to provide any useful information for which the Image is used (Computer Vision, Photography, Etc.,). The Contrast of the Image can be manipulated post processing the Image using ```Histogram Equalization``` based on Histograms. It uses a spatial function to operate on a pixel level. The Spatial function can bel called as operator or Transformation Function. 

```Histograms``` are defined as the pixel distribution of the image. It displays the frequency of a pixel in an Image.

### Properties of Transformation Function / Operator
- The function should be ```Single Valued``` and ```Monotonically Increasing``` from [0,1]
- ```Domain(T) = [0,1]``` & ```Range(T) = [0,1]```

## Histogram Equalization 

- Step 1 :  Compute Normalized Histogram 
```
def nHist(image):
    m, n = image.shape
    h = [0] * 256
    for i in range(m):
        for j in range(n):
            h[image[i,j]]+= 1

    return np.asarray(h) * 1.0 / (m * n)
```
- Step 2 : Compute Cumulative Distribution Function for Each bins in an Normalized Hitogram
```
def cdfunction(hist):
    cdfi = 0
    c = [0.0] * 256
    for i in range(len(hist)):
        cdfi+= hist[i]
        c[i] = cdfi 
    return np.array(c)
```
- Step 3 : Compute Transformation FUnction ```T = np.uint8(255 * cdf) ```
- Step 4 ; Transform the Pixel in an image based on the following relation ```EnhanceImage = T[Original Image]```

## RESULTS 

<p align="center"><img width=80% src="https://github.com/sudujr/Robotics-and-Visual-Navigation/blob/main/Contrast%20Limited%20Adaptive%20Histogram%20Equalization/InputandOutput/HE/NightVision_HE_IMAGES.jpg"></p>
<p align="center"><img width=80% src="https://github.com/sudujr/Robotics-and-Visual-Navigation/blob/main/Contrast%20Limited%20Adaptive%20Histogram%20Equalization/Histograms/HE/NightVision_HE_HIST.jpg></p>



