import cv2
import numpy as np

def calculate_optical_flow(prev_gray, gray, prev_pts, frame):
    pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, winSize=(15, 15),
                                               maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
