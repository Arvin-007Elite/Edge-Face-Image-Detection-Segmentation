import cv2

# Load the Image
image = cv2.imread('ASSETS\IMAGES\messi.jpeg')

# Convert the Image to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform Edge Detection using the Canny Algorithm
edges = cv2.Canny(gray, 100, 200)

# Display the Original Image and the Edges Detected
cv2.imshow('Original Image', image)
cv2.imshow('Edges Detected', edges)

# Wait for User Input to Exit the Program
cv2.waitKey(0)

# Release the Windows and Memory
cv2.destroyAllWindows()
