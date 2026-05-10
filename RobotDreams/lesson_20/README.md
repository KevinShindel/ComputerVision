# Distance to camera
- Camera calibration
- Focal length
- Bounding box size
- Object Size in px
- Zed Stereo Camera

# Data Augmentation
- Real close to real world ( production )
- Low Light
- Flip R-L
- Blur
- Only for training data ! ( test data should be real world )

# Data Disbalance Problem
- Add more data for low data classes
- Remove data for high data classes
- Data Augmentation for low data classes
- Class Weights ( loss function )
- Focal Loss ( loss function )
- Oversampling ( training data )

# ML model monitoring
- Statistics of input data ( mean, std, min, max )
- Statistics of output data ( mean, std, min, max )
- Performance metrics ( accuracy, precision, recall, F1-score )


# Cropping Image problem
- Padding + Resize
- Slicing + Resize (many iterations per image)
- Detector per slice ( remove slices without objects )
- Inference Slicer by Ultralytics ( remove slices without objects )

# Video-Stream pipeline
- Pipeline per Frame
- Latency ( time to process one frame )
- Batch processing ( process multiple frames at once )

# What next? 
- See transformers in Computer Vision ( ViT, Swin, etc. )
- See more about data augmentation ( Albumentations, etc. )
- See more about ML model monitoring ( TensorBoard, etc. )
- Stereo vision ( depth estimation, etc. )
- Kaggle, HuggingFace, RoboFlow for self-training 

# Anomaly Detection
- Variation Autoencoder ( VAE )

# HomeWork Feedback
> Great job and interesting presentation! <br/>
> This is a very well-structured and comprehensive project. <br/>
> You built a complete end-to-end pipeline for euro coin detection and went the extra mile by <br/> 
> implementing and comparing three completely different approaches: classical computer vision (Hough Circles), <br/>
> modern object detection (YOLO), and classification (TensorFlow CNN). <br/>
> Your repository is clean, and the README provides a very honest and insightful analysis of the models' strengths and <br/>
> weaknesses (such as YOLO confusing similar coins or the CNN struggling with angles). <br/>
> Extracting the total sum of the detected coins is also a really nice, practical touch ;-) <br/>
> Some thoughts for future improvements: To help the CNN recognize coins from different perspectives, <br/>
> some simple homographies or affine transforms can be added to your training (as augmentations). <br/>
> This will teach the model to recognize the elliptical shapes that circular coins take when viewed from an angle. <br/>
> I wish you the best of luck with your future computer vision projects! You have a great understanding of the field, <br/>
> so keep up the fantastic work! ;-)