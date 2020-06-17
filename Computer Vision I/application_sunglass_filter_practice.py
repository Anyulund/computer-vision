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

# plt.imshow(faceImage[:,:,::-1]);plt.title("Face")
# plt.show()

# Load the necklace image with Alpha channel
necklaceimagePath = os.path.join(DATA_PATH,"images/necklace2.png")
necklacePNG = cv2.imread(necklaceimagePath,-1)

#Resize the image to fit over the neck region
necklacePNG = cv2.resize(necklacePNG,(400,80))
print("image Dimension ={}".format(necklacePNG.shape))

# Separate the Color and alpha channels
necklaceBGR = necklacePNG[:,:,0:3]

necklaceMask1 = necklacePNG[:,:,3]
'''
# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(necklaceBGR[:,:,::-1]);plt.title('Necklace Color channels');
plt.subplot(122);plt.imshow(necklaceMask1,cmap = 'gray');plt.title('Necklace Alpha channel');
# plt.show()
'''
# Make a copy
faceWithNecklaceNaive = faceImage.copy()

# Replace the eye region with the sunglass image
faceWithNecklaceNaive[570:650,230:630]=necklaceBGR

# plt.imshow(faceWithNecklaceNaive[...,::-1])
# plt.show()

# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3 channel image for the mask
necklaceMask = cv2.merge((necklaceMask1, necklaceMask1, necklaceMask1))

# Make the values [0,1] since we are using arithmetic operations
necklaceMask = np.uint8(necklaceMask/255)

# Make a copy
faceWithNecklaceArithmetic = faceImage.copy()

#Get the eye region from the face image
neckROI = faceWithNecklaceArithmetic[570:650,230:630]

# Use the mask to create the masked neck region
maskedNeck = cv2.multiply(neckROI,(1- necklaceMask ))

#Use the mask to create the masked necklace region
maskedNecklace = cv2.multiply(necklaceBGR, necklaceMask)

#Conbine the Necklace in the neck region to get the augmented image
neckRoiFinal = cv2.add(maskedNeck, maskedNecklace)
'''
#Display the intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(maskedNeck[...,::-1]);plt.title("Masked Neck Region")
plt.subplot(132);plt.imshow(maskedNecklace[...,::-1]);plt.title("Masked Necklace Region")
plt.subplot(133);plt.imshow(neckRoiFinal[...,::-1]);plt.title("Augmented Neck and Necklace")
# plt.show()
'''
# Replace the neck ROI with the output form the code above
faceWithNecklaceArithmetic[570:650,230:630]=neckRoiFinal

#Display the final result
plt.figure(figsize=[20,20]);
plt.subplot(121);plt.imshow(faceImage[...,::-1]);plt.title("Original Image")
plt.subplot(122);plt.imshow(faceWithNecklaceArithmetic[...,::-1]);plt.title("With Necklace")
plt.show()
