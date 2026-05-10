# Tracking

## Frame Sequence
- Standard speed is 24 FPS ( minimum required )
- Eye buffer ( temp resolution of human vision, persistence of vision)

## Tracking VS Detection ( difference )
- Tracking is cheap, Detection is expensive
- Tracking working with detected objects
- Tracking is predictive by frame
- Tracking + Detection = Success !
- Run Detection on n-1 Frame where n - is variable from 1 to 25 ( for decrease loading by detector )
- Run Detection Once → Error Drift, Occlusion ( object loss by )
- Identity Preservations

## Tracking Model
- Motion Model - How objects move? Trajectory ?
- Appearance Model - How it looks like ?
- Optimization → Minimal loss by model

## Motion Model
- Prediction position 
- Probability of position
- Kalman Filters - Recursive Adaptive filter for tracking, who won trajectory prediction or detector ? 
- Implementations 
- - cv2.KalmanFilter - OpenCV implementation ( read documentation )
- - filterpy - Python library for Kalman filters ( read documentation )
- - stonesoup - Python library for tracking and state estimation ( read documentation )
- - pykalman - Python library for Kalman filters ( read documentation )

## Appearance Model
- Visual model description ( visualization probability)
- Block matching → Simple Matching Algorithm, Minimize Errors (SSD, SAD)

### Block matching problems
- Scaling ( moving speed )
- Background change ( contrast change )
- Different lighting ( tunnels, tree shadows etc.)


## Tracker Selection
- Long occlusions ? 
- If scale change ? 
- If object is change ? 
- Speed of moving ? 

## Common Trackers
| Tracker                          | Speed | CPU usage | When it makes sense to use                                                                                                                                  |
|----------------------------------|-------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CSRT (deprecated in 4.13.0)      | ⭐⭐    | ⭐⭐⭐⭐      | Best classical “default” when you need robustness to scale change and partial occlusion, and FPS is not tight                                               |
| KCF (deprecated in 4.13.0)       | ⭐⭐⭐⭐⭐ | ⭐⭐        | Real-time CPU tracking when appearance changes are small; weaker on heavy occlusion / fast motion / large scale change                                      |
| MIL (cv2.TrackerMIL)             | ⭐⭐⭐   | ⭐⭐⭐       | General single-object tracking with moderate appearance variation; can drift over time; okay when CSRT is too slow and KCF is too brittle                   |
| ViT (cv2.TrackerVit)             | ⭐     | ⭐⭐⭐⭐⭐     | Highest robustness under strong appearance changes if latency is acceptable; typically better with GPU / optimized builds                                   |
| Nano (cv2.TrackerNano)           | ⭐⭐⭐⭐  | ⭐⭐⭐       | Lightweight deep tracker: better robustness than classical trackers with still-reasonable CPU cost; good compromise for modern tracking on limited hardware |
| GOTURN (cv2.TrackerGOTURN)       | ⭐⭐⭐   | ⭐⭐⭐⭐      | Simple deep tracker when you have the model available; works for short-term tracking but weaker on long occlusions / major appearance changes               |
| DaSiamRPN (cv2.TrackerDaSiamRPN) | ⭐⭐⭐   | ⭐⭐⭐⭐      | Robust under fast motion and clutter; good speed/accuracy tradeoff, but still benefits from periodic re-detection for long occlusions / re-ID               |

> Do not use more frames which you have ( 50 > 25 )
> Few object detection by use detectors in parallel !

## Track by Descriptors ? 
- Use Face separate photo and separate video

## OpticalFlow
- OpticalFlow - is Relative motion between object and camera.
- [Kalman Filter](https://medium.com/@sophiezhao_2990/kalman-filter-explained-simply-2b5672429205)
- [Lucas-Kanade algorithm](https://dibyendu-biswas.medium.com/lucas-kanade-method-for-optical-flow-87ea48dd3e69) - is based on something known as Brightness constancy assumption. The key idea here is that pixel level brightness won’t change a lot in just one frame. It assumes that the color of an object does not change significantly and significantly in the previous two frames.
- [Gunnar-Farneback algorithm](https://www.geeksforgeeks.org/python/opencv-the-gunnar-farneback-optical-flow/) - is known as the pattern of apparent motion of objects, i.e, it is the motion of objects between every two consecutive frames of the sequence, which is caused by the movement of the object being captured or the camera capturing it
- [CV2 Documentation](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html)

| Flow Algorithm                      | Speed | CPU Usage | When it makes sense to use                                                                                                                                                                                           |
|-------------------------------------|-------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Lucas-Kanade                        | ⭐⭐⭐   | ⭐⭐        | Good for sparse tracking of distinctive features; can handle moderate motion but may struggle with large displacements or fast motion                                                                                |
| Gunnar-Farneback                    | ⭐⭐    | ⭐⭐⭐       | Better for dense flow estimation and handling larger motions; more computationally intensive, so better when you need detailed flow information and can afford the cost                                              |  
| Deep Learning-based (e.g., FlowNet) | ⭐     | ⭐⭐⭐⭐⭐     | Highest accuracy for complex motion patterns and large displacements; best when you have access to GPU acceleration and need state-of-the-art performance, but not suitable for real-time applications on CPU        |
| Kalman Filter                       | ⭐⭐⭐   | ⭐⭐⭐       | Not an optical flow method per se, but can be used in conjunction with flow estimates for improved tracking; good when you need to fuse multiple sources of information (e.g., flow + detection) for robust tracking |


## Recommended Materials

- [Tutorial on OpenCV trackers](https://docs.opencv.org/3.4/d2/d0a/tutorial_introduction_to_tracker.html)
- [Comprehensive demo](https://learnopencv.com/object-tracking-using-opencv-cpp-python/)
- [Tutorial on Kalman filters](https://kalmanfilter.net/)
- [Trackers overview](https://broutonlab.com/blog/opencv-object-tracking)
- [Lukas-Kanade optical flow](https://www.inf.fu-berlin.de/inst/ag-ki/rojas_home/documents/tutorials/Lucas-Kanade2.pdf)
- [trackers](https://github.com/roboflow/trackers) - Open-source implementations of various tracking algorithms, including deep learning-based trackers ( read documentation )
- [SuperVision Library](https://supervision.roboflow.com) - A Python library for video annotation and tracking, which includes implementations of various tracking algorithms and tools for visualizing tracking results ( read documentation )
- [YOLO v.11](https://docs.ultralytics.com/models/yolo11/) - The latest version of the YOLO (You Only Look Once) object detection model, which can be used in conjunction with tracking algorithms for improved performance ( read documentation )
- [HuggingFace Yolo](https://huggingface.co/Ultralytics/YOLO11) - A pre-trained version of the YOLO v.11 model available on Hugging Face, which can be easily integrated into tracking pipelines for object detection ( read documentation )
- [RoboFlow](https://roboflow.com/) - A platform for computer vision data management and model training, which provides tools for creating and managing datasets, training custom models, and deploying them for inference ( read documentation )

## Homework Feedback

Very good! There are plenty of other object tracking alternatives (besides OpenCV), e.g.,

- [roboflow](https://github.com/roboflow/trackers)
- [Ultralytics ](https://docs.ultralytics.com/modes/track/)
- [norfair](https://github.com/tryolabs/norfair)
- And don't forget the Kalman filter ;-)