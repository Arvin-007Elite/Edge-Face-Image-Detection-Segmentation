import cv2
import numpy as np

# Read the video
cap = cv2.VideoCapture('ASSETS\VIDEOS\Top 10 2011 Cricket World Cup catches.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Define the number of clusters (k) and criteria for k-means algorithm
k = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    
    if ret:
        # Reshape the frame to a 2D array of pixels and 3 color values (RGB)
        frame_reshaped = frame.reshape((-1,3))

        # Convert the data type to float32
        frame_reshaped = np.float32(frame_reshaped)

        # Apply k-means clustering algorithm
        _, labels, centers = cv2.kmeans(frame_reshaped, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert the centers back to uint8 data type and reshape to the original frame size
        centers = np.uint8(centers)
        segmented_frame = centers[labels.flatten()]
        segmented_frame = segmented_frame.reshape(frame.shape)

        # Write the segmented frame to output video
        out.write(segmented_frame)

        # Display the segmented frame
        cv2.imshow('Segmented Frame', segmented_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release the resources
cap.release()
out.release()
cv2.destroyAllWindows()
