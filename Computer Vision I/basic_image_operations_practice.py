#Import libraries
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

DATA_PATH ='/home/anna/Desktop/OpenCVcourse/Computer Vision I/'

#Read image
image = cv2.imread(os.path.join(DATA_PATH,"images/the-tooth-fairy.jpg"))

#Let's see what image we are dealing with:
plt.imshow(image[:,:,::-1])
plt.show()

#Create a new image by copying the already present image using the copy
imageCopy = image.copy()
plt.imshow(imageCopy[...,::-1])
plt.show()
'''
#Create an empty matrix
emptyMatrix = np.zeros((100,200,3),dtype = 'uint8')

plt.imshow(emptyMatrix)
plt.show()

#Feel the matrix with white Pixels

emptyMatrix = 255*np.ones((100,200,3),dtype='uint8')
plt.imshow(emptyMatrix)
plt.show()

#Create an empty matrix of the same size as original image
emptyOriginal = 100*np.ones_like(image)
plt.imshow(emptyOriginal)
plt.show()
'''
# Crop out a rectangle with a face
# x coordinates = 225 to 375
# y coordinates = 25 to 150
crop = image[25:150,225:375]
plt.imshow(crop[:,:,::-1])
plt.show()

#Put crop on the left and the right sides of the original picture
# Find height and width of the crop
cropHeight, cropWidth = crop.shape[:2]

# Copy to the left of the face
imageCopy[25:25+cropHeight,50:50+cropWidth] = crop
# Copy to the right of the face
imageCopy[25:25+cropHeight,425:425+cropWidth] = crop

# Display the output
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(image[...,::-1]);plt.title("Original Image");
plt.subplot(122);plt.imshow(imageCopy[...,::-1]);plt.title("Output Image");
plt.show()

#-------------------Resizing the image---------------------------------------

# -----Method 1 - Specify width and height
#Set rows and columns
resizeDownWidth = 400
resizeDownHeight = 300
resizedDown = cv2.resize(image,(resizeDownWidth,resizeDownHeight), interpolation= cv2.INTER_LINEAR)

#Mess up with aspect ratio
resizeUpWidth = 700
resizeUpHeight = 900
resizedUp = cv2.resize(image,(resizeUpWidth, resizeUpHeight),interpolation= cv2.INTER_LINEAR)

plt.figure(figsize=[15,15])
plt.subplot(131);plt.imshow(image[...,::-1]);plt.title("Original Image");
plt.subplot(132);plt.imshow(resizedDown[...,::-1]);plt.title("Resized Down Image");
plt.subplot(133);plt.imshow(resizedUp[...,::-1]);plt.title("Resized Up Image");
plt.show()

#-------Method 2 - Specify scaling factor
# Scaling Down the image 1.5 times by specifying both scaling factors
scaleUpX = 1.5
scaleUpY = 1.5

#Scaling Down the image 0.6 times specifying a single scale factor
scaleDown = 0.6

scaledDown = cv2.resize(image, None, fx=scaleDown, fy=scaleDown, interpolation = cv2.INTER_LINEAR)
scaledUp = cv2.resize(image, None, fx=scaleUpX, fy=scaleUpY, interpolation = cv2.INTER_LINEAR)

plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(scaledDown[...,::-1]);plt.title("Scaled Down Image");
plt.subplot(122);plt.imshow(scaledUp[...,::-1]);plt.title("Scaled Up Image");
plt.show()

#---------------Creating an Image Mask---------------------------------------
#------Create a mask using coordinates
# create an empty image of same size as the original
mask1 = np.zeros_like(image)
plt.imshow(mask1)
plt.show()

mask1[25:150,225:375] = 255
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(image[...,::-1]);plt.title("Original Image");
plt.subplot(122);plt.imshow(mask1[...,::-1]);plt.title("Mask");
plt.show()

#------Create a mask using pixel intensity or color
# Picking red as high intensity, green and blue are low intensity - BGR format
mask2 = cv2.inRange(image,(150,0,0),(255,100,100))
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(image[...,::-1]);plt.title("Original Image");
plt.subplot(122);plt.imshow(mask2[...,::-1]);plt.title("Red Masked Image");
plt.show()
