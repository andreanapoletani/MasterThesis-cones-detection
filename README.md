# Development of a visual-based tracking system for autonomous driving in Formula Student competition
The objective of this project is to develop a Computer Vision system capable of identifying the cones that delimit the track in driverless competitions in Formula Student.

The proposed system is developed starting from [yolov5](https://github.com/ultralytics/yolov5) project and is composed of two main phases:
- produce the best model using the original yolov5 training algorithm;
- optimize detection time with a custom version of yolov5 detection algorithm.

## Training phase

Different models have been produced with different choices in terms of hyperparameters and initial weights. 

## Detection optimization

The goal is to optimize detection times by using a Region Of Interest in which to search for cones within the frame. Through the use of a Kalman filter the position of the ROI is predicted in the next frame.
The idea is to predict the trajectory of the vehicle and therefore the position of the cones in the next reference frame.

![Immagine1](https://user-images.githubusercontent.com/48736255/160589541-576382c0-cc4d-4679-b43a-f4f13fee1dad.png)

## Documentation and references
See the full [thesis](https://www.linkedin.com/in/andrea-napoletani-aa0b87166/overlay/1635486254101/single-media-viewer/) for more details on the concepts applied in this project.

Andrea Napoletani @ 'Sapienza' Università di Roma. 

[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/andrea-napoletani-aa0b87166/)
&nbsp;
