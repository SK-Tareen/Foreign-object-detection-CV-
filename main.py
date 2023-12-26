# Import necessary libraries
import cv2
import imutils
import datetime
import numpy as np
import pandas as pd
import bg_model
# Initialize video capture, background model, and counters
cap = cv2.VideoCapture(0)  # Initialize video capture from default camera
avg = None  # Initialize variable for the background model
idx = 0  # Initialize index for image saving
idy = 0  # Unused variable
motionCounter = 0  # Counter to track continuous motion
minCounter = 20  # Minimum count to save images on continuous motion

# Create a window for displaying processed frames
cv2.namedWindow('Object_Detector', cv2.WINDOW_NORMAL)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),  # Window size for Lucas-Kanade optical flow
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Initialize variables for optical flow analysis
prev_gray = None  # Initialize previous frame in grayscale
prev_pts = None  # Initialize previous keypoints for optical flow

# Initialize x, y outside the loop
x, y = 0, 0  # Initialize coordinates for displaying text on the frame

# Main loop for video processing
while True:
    # Capture a frame and preprocess it
    ret, frame = cap.read()  # Read a frame from the video capture
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)  # Resize frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    gray = cv2.GaussianBlur(gray, (21, 21), 0)  # Apply Gaussian blur to reduce noise

    # Initialize background model if not done yet
    if avg is None:
        print("[INFO] starting background model...")
        avg = gray.copy().astype("float")  # Initialize the background model
        continue

    # Update the background model and calculate the frame difference
    cv2.accumulateWeighted(gray, avg, 0.7)  # Update the background model
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))  # Calculate frame difference

    # Threshold and dilate to obtain binary motion mask
    thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]  # Thresholding
    thresh = cv2.dilate(thresh, None, iterations=2)  # Dilate the thresholded image

    # Find contours in the motion mask
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    cnts = imutils.grab_contours(cnts)  # Extract contours using imutils

    # Loop through detected contours
    for c in cnts:
        # Filter out small contours
        if cv2.contourArea(c) < 10000:  # Set minimum contour area to consider
            continue

        # Draw bounding box and add text for detected object
        x, y, w, h = cv2.boundingRect(c)  # Get bounding box coordinates
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around the object
        text = "Foreign Object Detected"  # Text to display
        cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)  # Display text

        # Increment motion counter and save images on continuous motion
        motionCounter += 1  # Increment motion counter
        if motionCounter >= minCounter:  # Check if enough motion is detected
            idx += 1  # Increment image index
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Get current timestamp
            filename = f'C:/Users/hp/Desktop/FOD internship/time stamp/{timestamp}_{idx}.jpg'  # Create filename
            cv2.imwrite(filename, frame)  # Save the frame as an image
            motionCounter = 0  # Reset motion counter

    # Calculate optical flow and analyze motion
    if prev_gray is not None:
        pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)  # Calculate optical flow

        # Process optical flow results if available
        if pts is not None and prev_pts is not None:
            speeds = []  # List to store speeds
            directions = []  # List to store directions

            # Loop through keypoint pairs
            for i, (new, old) in enumerate(zip(pts, prev_pts)):
                a, b = new.ravel().astype(int)  # Coordinates of new point
                c, d = old.ravel().astype(int)  # Coordinates of old point

                # Calculate Euclidean distance between points as speed
                distance = np.sqrt((c - a) ** 2 + (d - b) ** 2)  # Calculate distance
                speed = distance  # Speed is considered as distance
                speeds.append(speed)  # Append speed to list

                # Calculate direction coordinates
                direction_x = c - a  # X-component of direction
                direction_y = d - b  # Y-component of direction
                directions.append((direction_x, direction_y))  # Append direction to list

                # Draw an arrow to show the optical flow direction
                mask = cv2.arrowedLine(frame, (a, b), (c, d), (0, 0, 255), 2)  # Draw arrow for motion direction

            # Calculate and display average speed
            if len(speeds) > 0:
                average_speed = np.mean(speeds)  # Calculate average speed
                cv2.putText(frame, f"Avg Speed: {average_speed:.2f} px/frame", (20, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # Display average speed

            # Display average direction coordinates
            if len(directions) > 0:
                direction_x, direction_y = np.mean(np.array(directions), axis=0)  # Calculate average direction
                cv2.putText(frame, f"Direction: ({direction_x:.2f}, {direction_y:.2f})", (20, frame.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # Display average direction

    # Display the processed frame and check for user input
    cv2.imshow("Object_Detector", frame)  # Display processed frame
    key = cv2.waitKey(1) & 0xFF  # Check for user input

    # Break the loop if 'q' is pressed
    if key == ord("q"):
        break

    # Update previous frame and keypoints for the next iteration
    prev_gray = gray.copy()  # Update previous frame in grayscale
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=100, qualityLevel=0.3,
                                       minDistance=7, blockSize=7)  # Update keypoints for optical flow

# Release video capture resources and close OpenCV windows
cap.release()  # Release video capture
cv2.destroyAllWindows()  # Close OpenCV windows


  
