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
'''
# Add the offset for increasing brightness
brightHigh = image + brightnessOffset

# Display the outputs
plt.figure(figsize=[20,20])
plt.subplot(121);plt.imshow(image[...,::-1]);plt.title("Original Image");
plt.subplot(122);plt.imshow(brightHigh[...,::-1]);plt.title("High Brightness");
plt.show();

print("Original Image Datatype : {}".format(image.dtype))
print("Brightness Image Datatype : {}".format(brightHigh.dtype))

print("Original Image Highest Pixel Intensity : {}".format(image.max()))
print("Brightness Image Highest Pixel Intensity : {}".format(brightHigh.max()))

a = np.array([[110,100],[120,130]],dtype='uint8')
print(a)

# Add 130 so that the last element encounters overflow
# print(a+130)
print(a-130)
print(a+(-130))

# Due to overflow and underflow problems, use OpenCV instead of Numpy to do mathematical operands
print(cv2.add(a,130))   #Clipping

# Convert to int32/int64 then clip
a_int32 = np.int32(a)
b = a_int32+130
print(b)

print(b.clip(0,255))
b_uint8 = np.uint8(b.clip(0,255))

#Convert to normalized float32/float64
a_float32 = np.float32(a)/255
b = a_float32 + 130/255
print(b)

c = b*255
print("Output = \n{}".format(c))
print("Clipped output= \n{}".format(c.clip(0,255)))
b_uint8 = np.uint8(c.clip(0,255))
print("uint8 output() = \n{}".format(b_uint8)
'''
#-----------------------Final Solution-----------------------------------------

brightnessOffset = 50

# Add the offset for increasing brightness
brightHighOpenCV = cv2.add(image, np.ones(image.shape,dtype='uint8')*brightnessOffset)

brightHighInt32 = np.int32(image) + brightnessOffset
brightHighInt32Clipped = np.clip(brightHighInt32,0,255)

# Display the outputs
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(image[...,::-1]);plt.title("original Image");
plt.subplot(132);plt.imshow(brightHighOpenCV[...,::-1]);plt.title("Using cv2.add function");
plt.subplot(133);plt.imshow(brightHighInt32Clipped[...,::-1]);plt.title("Using numpy and clipping");
plt.show()

# Add the offset for increasing brightness
brightHighFloat32 = np.float32(image) + brightnessOffset
brightHighFloat32NormalizedClipped = np.clip(brightHighFloat32/255,0,1)

brightHighFloat32ClippedUint8 = np.uint8(brightHighFloat32NormalizedClipped*255)

# Display the outputs
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(image[...,::-1]);plt.title("original Image");
plt.subplot(132);plt.imshow(brightHighFloat32NormalizedClipped[...,::-1]);plt.title("Using np.float32 and clipping");
plt.subplot(133);plt.imshow(brightHighFloat32ClippedUint8[...,::-1]);plt.title("Using int->float->int and clipping");
plt.show()
