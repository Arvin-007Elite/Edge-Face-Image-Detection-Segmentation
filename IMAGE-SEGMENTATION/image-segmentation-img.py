import cv2
import numpy as np

# Read the image
img = cv2.imread('ASSETS\IMAGES\car-race.jpeg')

# Reshape the image to a 2D array of pixels and 3 color values (RGB)
img_reshaped = img.reshape((-1,3))

# Convert the data type to float32
img_reshaped = np.float32(img_reshaped)

# Define the number of clusters (k) and criteria for k-means algorithm
k = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Apply k-means clustering algorithm
_, labels, centers = cv2.kmeans(img_reshaped, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert the centers back to uint8 data type and reshape to the original image size
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(img.shape)

# Display the input and segmented images
cv2.imshow('Input Image', img)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
