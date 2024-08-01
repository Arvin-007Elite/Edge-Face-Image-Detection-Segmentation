import cv2

# Load the Video Capture
cap = cv2.VideoCapture('ASSETS\VIDEOS\Dinesh Karthik hits 22 runs off Rubel Hossain - 19th over of Nidahas Trophy Final.mp4')

while cap.isOpened():
    # Read a Frame from the Video Capture
    ret, frame = cap.read()
    
    # Check if the video has ended
    if not ret:
        break

    # Convert the Frame to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform Edge Detection using the Canny Algorithm
    edges = cv2.Canny(gray, 100, 200)

    # Display the Edges Detected
    cv2.imshow('Edges Detected', edges)

    # Check for User Input to Exit the Program
    if cv2.waitKey(1) == 2:
        break

# Release the Video Capture and Destroy the Windows
cap.release()
cv2.destroyAllWindows()
