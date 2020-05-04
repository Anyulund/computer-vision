# Import libraries
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

DATA_PATH ='/home/anna/Desktop/OpenCVcourse/Computer Vision I/'

# Read image
image = cv2.imread(os.path.join(DATA_PATH,"images/the-tooth-fairy.jpg"))

#-------------------Datatype Conversion----------------------------------------
scalingFactor = 1/255.0

#Convert unsigned int to float
image = np.float32(image)
# Scale the values so that they lie between [0,1]
image = image*scalingFactor

#Convert back to unsigned int
image = image*(1.0/scalingFactor)
image = np.uint8(image)

#-------------------Contrast Enhancement----------------------------------------
'''
contrastPercentage = 30

# Multiply with scaling factor to increase contrast
contrastHigh = image*(1+contrastPercentage/100)

# Display the outputs
plt.figure(figsize=[20,20])
plt.subplot(121);plt.imshow(image[...,::-1]);plt.title("Original Image");
plt.subplot(122);plt.imshow(contrastHigh[...,::-1]);plt.title("High Contrast Image");
plt.show()

print("Original Image Datatype : {}".format(image.dtype))
print("Contrast Image Datatype : {}".format(contrastHigh.dtype))
print("Original Image Highest Pixel Intensity : {}".format(image.max()))
print("Contrast Image Highest Pixel Intensity : {}".format(contrastHigh.max()))

#Clip the values to [0, 255] and change it back to uint8 for display
contrastImage = image*(1+contrastPercentage/100)
clippedContrastImage = np.clip(contrastImage, 0, 255)
contrastHighClippedUint8 = np.uint8(clippedContrastImage)

#Convert the range to [0,1] and keep it in float format
contrastHighFloat = image*(1+contrastPercentage/100.0)
maxValue = image.max()
contrastHighNormalized01 = contrastHighFloat/maxValue

plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(image[...,::-1]);plt.title("Original Image");
plt.subplot(132);plt.imshow(contrastHighClippedUint8[...,::-1]); plt.title("Converted back to uint8");
plt.subplot(133);plt.imshow(contrastHighNormalized01[...,::-1]);plt.title("Normalized float to [0,1]");
plt.show()
'''
#---------------------Brightness Enhancement------------------------------------
brightnessOffset = 50

# Add the offset for increasing brightness
brightHigh = image + brightnessOffset

# Display the outputs
