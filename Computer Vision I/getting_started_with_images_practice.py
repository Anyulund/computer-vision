# Import Libraries
import cv2
import numpy as np
#just name data path later in the file for simplicity
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.figsize'] =(6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

DATA_PATH ='/home/anna/Desktop/OpenCVcourse/Computer Vision I/'
imagePath = os.path.join(DATA_PATH,"images/Hearts.jpg")

#-------------Part 1 - Image as a Matrix---------------
'''
#Read image in Grayscale format
testImage = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)

#Read image in Color format
#testImage = cv2.imread(imagePath,cv2.IMREAD_COLOR)

#Read image in Transparent format
#testImage = cv2.imread(imagePath,cv2.IMREAD_UNCHANGED)
print(testImage)
print("Data type = {}\n".format(testImage.dtype))
print("Object type = {}\n".format(type(testImage)))
print("Image Dimensions = {}\n.".format(testImage.shape))
'''

#--------------Part 2 - Manipulating Pixels ------------
'''
print(testImage[300,600])

testImage[0,0] = 0
print(testImage)

test_roi = testImage[0:2,0:4]
print("Original Matrix\n{}\n".format(testImage))
print("Selected Region\n{}\n".format(test_roi))

testImage[0:100,0:100] = 111
print("Modified Matrix\n{}\n".format(testImage))

plt.imshow(testImage)
plt.colorbar()
plt.show() # this line makes the plot show when called from command line

#Save image to disk
cv2.imwrite("Banana.jpg",testImage)

#-------------Part 3 - Color Images ---------------
colorimagePath = os.path.join(DATA_PATH,"images/the-tooth-fairy.jpg")

#Read the image
img = cv2.imread(colorimagePath)
print("image Dimension ={}".format(img.shape))

#Display image
plt.imshow(img)
plt.title("Original Image")
plt.show()

#Convert BGR to RGB colorspace
imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(imgRGB)
plt.title("BGR to RGB conversion")
plt.show()

#Reverse channels
plt.imshow(img[:,:,::-1])
plt.title("Channel reversal")
plt.show()

#Show the channels
plt.figure(figsize=[20,5])

plt.subplot(131);plt.imshow(img[:,:,0]);plt.title("Blue Channel");
plt.subplot(132);plt.imshow(img[:,:,1]);plt.title("Green Channel");
plt.subplot(133);plt.imshow(img[:,:,2]);plt.title("Red Channel");
plt.show()

#-----------Part 3.1 - Splitting and Merging channels

#Split the image into the B,G,R components
b,g,r = cv2.split(img)

#Show the channels
plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(b);plt.title("Blue Channel");
plt.subplot(142);plt.imshow(g);plt.title("Green Channel");
plt.subplot(143);plt.imshow(r);plt.title("Red Channel");

plt.show()

#---------Part 3.2 - Manipulating Color Pixels
testImage = cv2.imread(imagePath,1)
plt.imshow(testImage)
plt.show()

#Access Color Pixel
print(testImage[0,0])

plt.figure(figsize=[20,20])

#Yellow = Red + Green
testImage[0,0] = (0,255,255)
plt.subplot(131);plt.imshow(testImage[:,:,::-1])

#Cyan = Blue + Green
testImage[0,0] = (255,255,0)
plt.subplot(132);plt.imshow(testImage[:,:,::-1])

#Magenta = Red + Blue
testImage[0,0] = (255,0,255)
plt.subplot(133);plt.imshow(testImage[:,:,::-1])

plt.show()

#Modify Region of Interest
testImage[0:100,0:100]   = (255,0,0)
testImage[100:200,0:100] = (0,255,0)
testImage[200:300,0:100] = (0,0,255)

plt.imshow(testImage[:,:,::-1])
plt.show()
'''

#---------Part 3.3 - Images with Alpha Channel
pngimagePath = os.path.join(DATA_PATH,"images/Polar-Bear.png")

#Read the image
#Note that we are passing flag = -1 while reading the image (it will read the image as is)
imgPNG = cv2.imread(pngimagePath, -1)
imgRGB = cv2.cvtColor(imgPNG, cv2.COLOR_BGR2RGB)
plt.imshow(imgRGB)
plt.show()
print("image Dimension ={}".format(imgPNG.shape))
#First 3 channels will be combined to form BGR image
#Mask is the alpha channel of the original image
imgBGR = imgPNG[:,:,0:3]
imgMask = imgPNG[:,:,3]
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(imgBGR[:,:,::-1]);plt.title('Color channels');
plt.subplot(122);plt.imshow(imgMask,cmap='gray');plt.title('Alpha channel');
plt.show()
