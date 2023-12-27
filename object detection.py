import cv2
import datetime

def detect_objects(cnts, frame, motionCounter, minCounter, idx):
    for c in cnts:
        if cv2.contourArea(c) < 10000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Foreign Object Detected"
        cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        motionCounter += 1
        if motionCounter >= minCounter:
            idx += 1
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'C:/Users/hp/Desktop/FOD internship/time stamp/{timestamp}_{idx}.jpg'
            cv2.imwrite(filename, frame)
            motionCounter = 0
