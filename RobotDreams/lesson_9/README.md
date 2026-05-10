# Face Detection by ML


## Robust Identification is:

- Universal feature
- Unique feature
- Measurable feature
- Accurate feature


## Base workflow for face detection:

1. **Face detection** - find pattern on photo
2. **Face alignment** - extract face and align it ( rotate / scale , etc.)
3. **Face recognition** - Identify face


## Haar binary functions
- Sliding window 24x24px
- Four features per window
- [Wiki page](https://en.wikipedia.org/wiki/Haar-like_feature)


## Viola-Jones Object Detector

- Features - 
- Integral image - calculating CumSum of all pixels on image
- AdaBoost - lot of weak classificators for predict probability of face
- Cascading - 160k features per window 24x24px ( sort filters by power, skip window when probability lower than threshold )
- [Wiki Page](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework)
- Front-face recognition only! ( Banks / Phone ID etc.) 
- Doesn't work on surveillance cameras / side photos etc...

## Dlib Face Detector
- Models [GitHub Link](https://github.com/davisking/dlib-models)
- Use dlib library
- Very Slow Performance
- Minimum faces 80x80
- Smaller Bounding Box

## OpenCV DNN
- Models [GitHub link](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
+ Fast and accurate
+ Works well on small images < 80x80px


## Facial Alignment

- Face keypoints ( years, eyes, nose etc.)

## Recommended links

- [Face detection with dlib and OpenCV](https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/) <- This usefull for Homework
- [Viola-Jones explained](https://towardsdatascience.com/viola-jones-algorithm-and-haar-cascade-classifier-ee3bfb19f7d8/)
- [Tutorial on Haar cascades](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
- [Precision vs Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
- [Haar cascade training tutorial](https://docs.opencv.org/3.4/dc/d88/tutorial_traincascade.html)

## Homework Feedback

Dobra večer, Tymure!

Apart from the Deepface library that we already mentioned in the lecture ( https://viso.ai/computer-vision/deepface/ ), you can also checkout some huggingface models, e.g., https://huggingface.co/arnabdhar/YOLOv8-Face-Detection or https://huggingface.co/qualcomm/MediaPipe-Face-Detection