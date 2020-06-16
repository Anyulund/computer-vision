import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

DATA_PATH ='/home/anna/Desktop/OpenCVcourse/Computer Vision I/'

# Load the Face Image
faceImagePath = os.path.join(DATA_PATH,"images/anna.png")
faceImage = cv2.imread(faceImagePath)

plt.imshow(faceImage[:,:,::-1]);plt.title("Face")
# plt.show()

# Load the necklace image with Alpha channel
necklaceimagePath = os.path.join(DATA_PATH,"images/necklace2.png")
necklacePNG = cv2.imread(necklaceimagePath,-1)

#Resize the image to fit over the neck region
necklacePNG = cv2.resize(necklacePNG,(320,50))
print("image Dimension ={}".format(necklacePNG.shape))

# Separate the Color and alpha channels
necklaceBGR = necklacePNG[:,:,0:3]
necklaceMask = necklacePNG[:,:,3]

# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(necklaceBGR[:,:,::-1]);plt.title('Necklace Color channels');
plt.subplot(122);plt.imshow(necklaceMask,cmap = 'gray');plt.title('Necklace Alpha channel');
# plt.show()


# Make a copy
faceWithNecklaceNaive = faceImage.copy()

# Replace the eye region with the sunglass image
faceWithNecklaceNaive[570:620,270:590]=necklaceBGR

plt.imshow(faceWithNecklaceNaive[...,::-1])
# plt.show()
