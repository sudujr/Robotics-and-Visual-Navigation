
# Import Necessary Module
import cv2
import numpy as np

# Load the original image to get the height and weight for reconstruction
image = cv2.imread("assignment_1.jpg")
h,w,c = image.shape
print(h,w)
# open assignment_1_cartesian_coords.txt in read mode
file = open("assignment_1_cartesian_coords.txt","r")
c = 0
col = []
row = []
#Convert the coordinates into rows and column of image
for f in file:
    x = f.split(",")
    col.append(int(x[0]))
    row.append(-1 * int(x[1].strip()))
    c += 1
x = max(row)
y = max(col)
# print(row[0])
# print(col[0])
print(x)
print(y)
# print(len(row))
# print(len(col))
file.close()

file1 = open("RGB_values.txt","r")
val = []
for f in file1:
    print(f)
    z = ' '.join(f.split())
    z = z[1:-1].strip()
    z = z.split(" ")
    z =[int(i) for i in z]
    val.append(z)
file1.close()

i = 0
r = np.empty([h,w], dtype=np.uint8)
g = np.empty([h,w], dtype=np.uint8)
b = np.empty([h,w], dtype=np.uint8)
while i < len(row):
    image[row[i]][col[i]][0] = val[i][0]
    image[row[i]][col[i]][1] = val[i][1]
    image[row[i]][col[i]][2] = val[i][2]

    i += 1



cv2.imwrite('r.jpg', image)
cv2.imshow("image", image)
cv2.waitKey(0)










