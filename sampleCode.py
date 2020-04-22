# Import modules
import cv2

# Read image in GrayScale mode
image = cv2.imread("boy.jpg", 0)

# Write the grayscale image
cv2.imwrite("boyGray.jpg",image)
