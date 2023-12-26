# Import necessary libraries
import cv2
import imutils
import datetime
import numpy as np
import pandas as pd

# Initialize video capture, background model, and counters
cap = cv2.VideoCapture(0)  # Initialize video capture from default camera
avg = None  # Background model initialization
idx = 0  # Index for saved image filenames
idy = 0  # Unused variable
motionCounter = 0  # Counter to track continuous motion
minCounter = 20  # Minimum count required for continuous motion detection

# Create a window for displaying processed frames
cv2.namedWindow('Object_Detector', cv2.WINDOW_NORMAL)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize variables for optical flow analysis
prev_gray = None  # Store previous frame in grayscale
prev_pts = None  # Previous keypoints for optical flow analysis

# Initialize x, y outside the loop
x, y = 0, 0  # Variables to store bounding box coordinates

# Main loop for video processing
while True:
    # Capture a frame and preprocess it
    ret, frame = cap.read()  # Read a frame from the video capture
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)  # Resize the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    gray = cv2.GaussianBlur(gray, (21, 21), 0)  # Apply Gaussian blur to reduce noise

    # Initialize background model if not done yet
    if avg is None:
        print("[INFO] starting background model...")
        avg = gray.copy().astype("float")
        continue

    # Update the background model and calculate the frame difference
    cv2.accumulateWeighted(gray, avg, 0.7)  # Update the background model
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))  # Calculate frame difference

    # Threshold and dilate to obtain binary motion mask
    thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]  # Apply thresholding
    thresh = cv2.dilate(thresh, None, iterations=2)  # Dilate the thresholded image

    # Find contours in the motion mask
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)  # Grab contours using imutils

    # Loop through detected contours
    for c in cnts:
        # Filter out small contours
        if cv2.contourArea(c) < 10000:
            continue

        # Draw bounding box and add text for detected object
        x, y, w, h = cv2.boundingRect(c)  # Get bounding box coordinates
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
        text = "Foreign Object Detected"
        cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)  # Add text

        # Increment motion counter and save images on continuous motion
        motionCounter += 1
        if motionCounter >= minCounter:
            idx += 1
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'C:/Users/hp/Desktop/FOD internship/time stamp/{timestamp}_{idx}.jpg'
            cv2.imwrite(filename, frame)  # Save the frame as an image
            motionCounter = 0  # Reset motion counter

    # Calculate optical flow and analyze motion
    if prev_gray is not None:
        pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

        # Process optical flow results if available
        if pts is not None and prev_pts is not None:
            speeds = []  # Store calculated speeds
            directions = []  # Store calculated directions

            # Loop through keypoint pairs
            for i, (new, old) in enumerate(zip(pts, prev_pts)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)

                # Calculate Euclidean distance between points
                distance = np.sqrt((c - a) ** 2 + (d - b) ** 2)

                # Calculate speed (distance / time)
                speed = distance  # In this case, speed is represented by distance
                speeds.append(speed)  # Append calculated speed to the list

                # Calculate direction coordinates
                direction_x = c - a
                direction_y = d - b
                directions.append((direction_x, direction_y))  # Append direction coordinates

                # Draw an arrow to show the optical flow direction
                mask = cv2.arrowedLine(frame, (a, b), (c, d), (0, 0, 255), 2)  # Draw arrowed line

            # Calculate and display average speed
            if len(speeds) > 0:
                average_speed = np.mean(speeds)
                cv2.putText(frame, f"Avg Speed: {average_speed:.2f} px/frame", (20, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # Add text for average speed
            
            # Display average direction coordinates
            if len(directions) > 0:
                direction_x, direction_y = np.mean(np.array(directions), axis=0)
                cv2.putText(frame, f"Direction: ({direction_x:.2f}, {direction_y:.2f})", (20, frame.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # Add text for average direction

    # Display the processed frame and check for user input
    cv2.imshow("Object_Detector", frame)  # Display the processed frame
    key = cv2.waitKey(1) & 0xFF  # Wait for key press

    # Break the loop if 'q' is pressed
    if key == ord("q"):
        break

    # Update previous frame and keypoints for the next iteration
    prev_gray = gray.copy()  # Update previous frame in grayscale
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=100, qualityLevel=0.3,
                                       minDistance=7, blockSize=7)  # Update keypoints for optical flow

# Release video capture resources and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

            

  
