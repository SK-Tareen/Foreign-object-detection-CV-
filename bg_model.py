# background_model.py
import cv2

def initialize_background_model(gray):
    avg = gray.copy().astype("float")
    return avg

def update_background_model(avg, gray):
    cv2.accumulateWeighted(gray, avg, 0.7)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    return frameDelta
