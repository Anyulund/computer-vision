# Import Libraries
import cv2
import numpy as np
#just name data path later in the file for simplicity
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.figsize'] =(6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

DATA_PATH ='/home/anna/Desktop/OpenCVcourse/Computer Vision I/images/Hearts.jpg'


#-------------Part 1 - Image as a Matrix---------------

#Read image in Grayscale format
testImage = cv2.imread(DATA_PATH,cv2.IMREAD_GRAYSCALE)
'''
#Read image in Color format
#testImage = cv2.imread(DATA_PATH,cv2.IMREAD_COLOR)

#Read image in Transparent format
#testImage = cv2.imread(DATA_PATH,cv2.IMREAD_UNCHANGED)
print(testImage)
print("Data type = {}\n".format(testImage.dtype))
print("Object type = {}\n".format(type(testImage)))
print("Image Dimensions = {}\n.".format(testImage.shape))
'''

#--------------Part 2 - Manipulating Pixels ------------
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
