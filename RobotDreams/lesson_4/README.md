# Symmetrical Filter (Prewitt) for Edge Detection

```python
import cv2
import numpy as np

img = np.ndarray([])

h_kernel = [[-1 -1 -1], [0, 0, 0], [1, 1,1]]
v_kernel = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]

img_hf = cv2.filter2D(img, ddepth=1, kernel=h_kernel)
img_vf = cv2.filter2D(img, ddepth=1, kernel=v_kernel)
```

# Averaging Filter (Sobol Filter)

```python
import cv2
import numpy as np

img = np.ndarray([])

h_kernel = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
v_kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

img_hf = cv2.filter2D(img, ddepth=1, kernel=h_kernel)
img_vf = cv2.filter2D(img, ddepth=1, kernel=v_kernel)
```


# Canny Edge Detector

1. Noise Reduction (Gaussian Filter)
2. Gradient Calculation (Sobel Filter)
3. Non-maximum Suppression (NMS)
4. Double Thresholding (Hysteresis Thresholding) Th + Tl ( G> Th = edge, G < Tl = No Edge, Tl < G < Th = Potential Edge)
5. Edge Tracking by Hysteresis

# Hough Transformation
- Voting Algorithm
- TODO: Found Function in cv2 

## Homework Feedback
- Very good! In practice we don't process each video frame separately but we also use temporal information, i.e.,
- the detections from previous frames. 
- By combining detections from several frames we can stabilize the output and obtain more accurate detections.