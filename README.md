# Foreign-object-detection-CV
Project by: 
1. Sania Khan Tareen 332190
2. Maheen Salman 342603
3. Minahil Shafqat 331805
## Abstract
This project implements a real-time motion detection system using Python and OpenCV, designed for surveillance and security applications. The system captures video frames from a camera, establishes a dynamic background model, and identifies areas with significant motion by calculating the difference between consecutive frames. Detected motion triggers the drawing of bounding boxes around the moving objects and the display of a descriptive text label. The project provides an option to save snapshots of the detected motion, each file named with a timestamp and an index. 

The application demonstrates the fundamental principles of motion detection and serves as a foundation for more advanced security and surveillance solutions. The system is configurable, allowing users to adjust parameters such as motion sensitivity and the minimum duration of detected motion required to trigger snapshot capture. Overall, this project offers a versatile and customizable motion detection tool for various security and monitoring scenarios.
## Methodology
In this project, we use OpenCV and  acquire frames from any video. Each frame  is used to discern and accentuate foreign objects. We monitor consecutive frames with identified motion and use a 'minCounter' to establish the minimum number of consecutive frames necessary to initiate object detection.

The initiation of the background model involves the accumulation of a weighted average of grayscale frames, with continuous updates as frames are processed. Motion detection is computed by taking the absolute difference between the current frame and the background model, followed by thresholding the delta image and dilating the thresholded image to highlight regions with detected motion. In this way, we identify contours  and draw corresponding bounding boxes labeled "Foreign Object Detected". 

The project includes a provision to save snapshots of frames with detected foreign objects in JPEG format, contingent upon the motion counter surpassing the predefined minimum threshold ('minCounter'). The processed frames, featuring bounding boxes and descriptive text, are presented in real-time. 

Ultimately, the release of the video capture stream and the closure of OpenCV windows conclude the project. Parameter adjustments, such as minimum contour area and dilate iterations, can be tailored to specific project requisites. Additionally, ensuring the validity of the file path for saving detected objects is imperative for project functionality.
                                                ![image](https://github.com/SK-Tareen/Foreign-object-detection-CV-/assets/87571978/6861c63f-8c43-41a7-927a-4e8a1cbc2771)

