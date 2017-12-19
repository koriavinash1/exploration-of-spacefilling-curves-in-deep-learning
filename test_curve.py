#this code shows an example implementation of hilbert operations on image

import matplotlib.pyplot as plt
import numpy as np
import cv2, os
from hilbert_ops import image2signal, signal2image

# load image
image = cv2.imread("./images/11.jpg")
cv2.waitKey(10)

image = cv2.resize(image, (32,32), interpolation=cv2.INTER_AREA)
# image = cv2.cvtColor(image,  cv2.COLOR_BGR2GRAY)
# ret, image = cv2.threshold(image, 127,255,cv2.THRESH_BINARY)

hilbert_coordinates = np.arange(image.shape[0]*image.shape[1])
himage = image2signal(image)
print himage.shape
Rimage = signal2image(himage)

# print himage
# plt.plot(hilbert_coordinates, himage[0])
# plt.plot(hilbert_coordinates, himage[0])
# plt.plot(hilbert_coordinates, himage[2])
# plt.show()

plt.imshow(Rimage.reshape(32,32,3), cmap="gray")
plt.show()