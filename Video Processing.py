import cv2  # Import the OpenCV library for computer vision tasks
import imutils  # Import imutils for convenient image processing functions
import datetime  # Import datetime for working with date and time
import numpy as np  # Import numpy for numerical operations

# Function to initialize the capture from the camera
def initialize_capture():
    cap = cv2.VideoCapture(0)  # Access the default camera (index 0)
    return cap  # Return the video capture object

# Function to preprocess a frame from the video feed
def preprocess_frame(frame):
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    # Resize the frame to a fixed size of 640x480 pixels using linear interpolation
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the grayscale frame to reduce noise
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    return gray  # Return the preprocessed grayscale frame
