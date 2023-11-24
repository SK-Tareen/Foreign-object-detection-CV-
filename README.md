# Foreign-object-detection-CV
## Abstract
This project implements a real-time motion detection system using Python and OpenCV, designed for surveillance and security applications. The system captures video frames from a camera, establishes a dynamic background model, and identifies areas with significant motion by calculating the difference between consecutive frames. Detected motion triggers the drawing of bounding boxes around the moving objects and the display of a descriptive text label. The project provides an option to save snapshots of the detected motion, each file named with a timestamp and an index. 

The application demonstrates the fundamental principles of motion detection and serves as a foundation for more advanced security and surveillance solutions. The system is configurable, allowing users to adjust parameters such as motion sensitivity and the minimum duration of detected motion required to trigger snapshot capture. Overall, this project offers a versatile and customizable motion detection tool for various security and monitoring scenarios.
## Methodology
In this object detection project using OpenCV, the methodology involves capturing a video feed from a webcam and processing each frame to identify and highlight foreign objects. 

The project initializes variables such as `avg` for the background model, `idx` for image file indexing, `motionCounter` for tracking consecutive frames with detected motion, and `minCounter` to set the minimum number of consecutive frames required for triggering object detection. The background model is initiated by accumulating the weighted average of grayscale frames, and this model is continuously updated as frames are processed. 

The motion detection process involves computing the absolute difference between the current frame and the background model, thresholding the delta image, and dilating the thresholded image to emphasize regions with motion. Identified contours are filtered based on their area, and bounding boxes are drawn around larger contours labeled as "Foreign Object Detected." 

Frames with detected foreign objects are saved as JPEG images if the motion counter exceeds the defined minimum (`minCounter`). The processed frames, including bounding boxes and text, are displayed in real-time. User interaction is considered by checking for the 'q' key to break the processing loop. Finally, the video capture stream is released, and OpenCV windows are closed. 
Adjustments to parameters, such as minimum contour area and dilate iterations, can be made based on specific project requirements. Additionally, it is essential to ensure the validity of the file path for saving detected objects.
