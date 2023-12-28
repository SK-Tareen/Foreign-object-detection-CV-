import cv2
import imutils
import datetime
import numpy as np

def initialize_capture():
    cap = cv2.VideoCapture(0)
    return cap

def preprocess_frame(frame):
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray
