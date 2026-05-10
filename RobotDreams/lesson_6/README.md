from sys import flags

# KeyPoints

- TODO: describe this point
- Edges on object of interest

# KeyPoints Properties

- Accurate localization
- Invariance against shift/scale/rotation/brightness changes
- Robust Against Noise

# Corner Detector

- Harris detector


# Harris corner detector

- Flat region: no changes in all directions
- Edge: no changes along edge direction
- Corner: significant change in all directions

```python
import numpy as np
import cv2

img = np.ndarray([])
dst = cv2.cornerHarris(img, # image
                       blockSize=2,
                       ksize=11,
                       k=0.04
                       )
```

| Parameter | Value | Description                                                                                          |
|-----------|-------|------------------------------------------------------------------------------------------------------|
| blockSize | 4     | more sensitive to fine detail/noise => more corners, but less stable                                 |
| blockSize | 20    | more stable corners, less noise sensitivity, but poorer localization and missed small corners.       |
| ksize     | 3     | sharper gradients, more sensitive to noise, better localization.                                     | 
| ksize     | 7     | smoother gradients, less noise, but corners become “broader” and potentially less precisely located. |
| k         | 0.04  | more corner-like responses prioritized (often more detections).                                      | 
| k         | 0.06  | stricter; may suppress some corners and respond differently on edges.                                | 


# SIFT corner detector ( Scale Invariant Feature Transform)

- Calculate max points
- Calculate as Harris algorithm
- Calculate descriptor of points

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = np.ndarray([])
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray, None) # uint8 expected !

out = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(14, 8)), plt.imshow(out), plt.grid(False)
```

## Recommended materials

1. [SIFT original paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
2. [OpenCV tutorial for Harris detector](https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html)
3. [Comparison SIFT vs SURF vs ORB](https://shehan-a-perera.medium.com/a-comparison-of-sift-surf-and-orb-333d64bcaaea)
4. [Object recognition with SIFT](https://www.cs.princeton.edu/courses/archive/spr08/cos598B/Lectures/SIFT%20and%20Object%20Recognition.pdf)


## Homework Feedback
Well done! And if we see that the image resolution is too large,
we can always resize the image to a smaller size (this applies to many other problems in CV, not only here) ;-)