# Filters edge detection 

<hr/>

## Edge Detection Filters

| Filter / Method                       | cv2 / NumPy Implementation                                                            | When to Use                                                                                        | When NOT to Use                                                                   | Description                                                                                                                            |
| ------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Canny Edge Detection**              | `cv2.Canny(image, threshold1, threshold2)`                                            | General-purpose edge detection; object contours; segmentation; shape detection                     | Very noisy images without preprocessing; when edge intensity varies significantly | Multi-stage edge detector using Gaussian smoothing, gradient calculation, non-maximum suppression, double thresholding, and hysteresis |
| **Sobel X**                           | `cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)`                                         | Detecting **vertical edges**; measuring horizontal intensity changes                               | When you need orientation-independent edges                                       | Computes the first derivative of image intensity in the X direction                                                                    |
| **Sobel Y**                           | `cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)`                                         | Detecting **horizontal edges**; measuring vertical intensity changes                               | When you need orientation-independent edges                                       | Computes the first derivative of image intensity in the Y direction                                                                    |
| **Sobel Magnitude**                   | `gx = cv2.Sobel(...,1,0); gy = cv2.Sobel(...,0,1); magnitude = cv2.magnitude(gx, gy)` | General gradient magnitude; feature extraction; edge strength estimation                           | When you need thin, well-localized edges directly                                 | Combines X/Y gradients to estimate the strength of local intensity changes                                                             |
| **Scharr X**                          | `cv2.Scharr(image, cv2.CV_64F, 1, 0)`                                                 | High-quality detection of vertical gradients; small kernels; better rotational accuracy than Sobel | When computational simplicity is more important than gradient accuracy            | Improved Sobel operator optimized for a 3×3 kernel                                                                                     |
| **Scharr Y**                          | `cv2.Scharr(image, cv2.CV_64F, 0, 1)`                                                 | High-quality detection of horizontal gradients                                                     | When you need a larger smoothing/derivative kernel                                | Improved derivative operator with better rotational symmetry than 3×3 Sobel                                                            |
| **Laplacian**                         | `cv2.Laplacian(image, cv2.CV_64F)`                                                    | Detecting rapid intensity changes in **all directions**; blob/edge analysis                        | Noisy images without smoothing; when edge orientation is required                 | Second-order derivative operator; responds to intensity changes regardless of direction                                                |
| **Roberts Cross**                     | `numpy` convolution with `[[1,0],[0,-1]]` and `[[0,1],[-1,0]]`                        | Very simple edge detection; lightweight processing                                                 | Noisy images; high-quality CV applications; low-resolution images                 | Small 2×2 gradient operator that detects diagonal intensity changes                                                                    |
| **Prewitt X**                         | `cv2.filter2D(image, -1, np.array([[-1,0,1],[-1,0,1],[-1,0,1]]))`                     | Simple vertical edge detection; educational implementations                                        | Noisy images; applications requiring precise gradients                            | First-order derivative operator similar to Sobel but without Sobel's weighting                                                         |
| **Prewitt Y**                         | `cv2.filter2D(image, -1, np.array([[-1,-1,-1],[0,0,0],[1,1,1]]))`                     | Simple horizontal edge detection                                                                   | Noisy images; applications requiring high-quality gradient estimation             | Detects horizontal intensity transitions using a 3×3 kernel                                                                            |
| **Prewitt Magnitude**                 | `cv2.filter2D()` + `np.sqrt(gx**2 + gy**2)`                                           | Simple orientation-independent edge detection                                                      | When robustness to noise is important                                             | Combines Prewitt X/Y gradients into an edge-strength map                                                                               |
| **Difference of Gaussian (DoG)**      | `cv2.GaussianBlur()` + `cv2.subtract()`                                               | Multi-scale edge/blob detection; removing low-frequency illumination                               | Precise localization of individual edges; very noisy images                       | Subtracts two Gaussian-blurred images with different σ values to emphasize spatial transitions                                         |
| **Gaussian Gradient**                 | `cv2.GaussianBlur()` → `cv2.Sobel()`                                                  | Images with moderate noise where smoother gradients are required                                   | Very fine/small edges that may disappear during smoothing                         | Gaussian smoothing followed by gradient computation                                                                                    |
| **Morphological Gradient**            | `cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)`                                 | Binary/mask boundaries; object contours; segmentation results                                      | Natural images where intensity-based edges are required                           | Difference between morphological dilation and erosion, producing object boundaries                                                     |
| **Laplacian of Gaussian (LoG)**       | `cv2.GaussianBlur()` → `cv2.Laplacian()`                                              | Noise reduction + second-order edge detection; blob/structure analysis                             | When you require Canny-like thin and connected edges                              | Smooths the image with Gaussian filtering and then applies the Laplacian                                                               |
| **Zero-Crossing LoG**                 | `cv2.GaussianBlur()` + `cv2.Laplacian()` + `np.sign()` / neighborhood analysis        | Detecting precise transitions using second-derivative zero crossings                               | Noisy images; applications requiring robust edge connectivity                     | Finds locations where the Laplacian changes sign, indicating possible edges                                                            |
| **Adaptive Threshold Boundary**       | `cv2.adaptiveThreshold()` + morphological operations                                  | Uneven illumination; document/object segmentation                                                  | Natural scenes where continuous gradient information is required                  | Converts local intensity differences into a binary boundary/foreground representation                                                  |
| **Binary Threshold Boundary**         | `cv2.threshold()` + `cv2.morphologyEx(..., cv2.MORPH_GRADIENT, ...)`                  | High-contrast objects with relatively uniform illumination                                         | Variable illumination; low-contrast boundaries                                    | Creates a binary representation and extracts its morphological boundaries                                                              |
| **Contour Detection**                 | `cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`                | Finding object boundaries after segmentation; shape analysis                                       | Directly on noisy grayscale images without segmentation                           | Extracts connected object boundaries from binary images                                                                                |
| **Canny + Contours**                  | `edges = cv2.Canny(...); contours, _ = cv2.findContours(edges, ...)`                  | Object detection based on shape; geometric measurements                                            | Images where Canny produces many fragmented edges                                 | Combines gradient-based edge detection with contour extraction                                                                         |
| **Hough Line Transform**              | `cv2.HoughLines(edges, rho, theta, threshold)`                                        | Detecting straight lines; roads, document borders, structures                                      | Curved boundaries; highly fragmented/noisy edges                                  | Transforms edge pixels into parameter space to identify straight lines                                                                 |
| **Probabilistic Hough Lines**         | `cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength, maxLineGap)`            | Detecting actual line segments efficiently                                                         | Curved boundaries; very weak edges                                                | Faster Hough-based method that returns finite line segments                                                                            |
| **Hough Circle Transform**            | `cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, ...)`                                    | Detecting circular boundaries; wheels, coins, round objects                                        | Non-circular objects; highly elliptical/irregular shapes                          | Uses gradient information and voting to detect circular structures                                                                     |
| **Morphological Boundary Extraction** | `boundary = cv2.subtract(image, cv2.erode(image, kernel))`                            | Extracting boundaries from binary masks; segmentation post-processing                              | General-purpose edge detection in natural images                                  | Obtains the boundary by subtracting an eroded image from the original                                                                  |
| **NumPy Gradient**                    | `gx = np.gradient(image, axis=1); gy = np.gradient(image, axis=0)`                    | Simple numerical gradient; custom algorithms; prototyping                                          | Production CV pipelines requiring optimized filtering                             | Computes numerical derivatives directly using NumPy                                                                                    |
| **NumPy Finite Difference**           | `gx = image[:, 2:] - image[:, :-2]`                                                   | Custom derivative operators; research/prototyping                                                  | When precise kernel control and robust noise handling are required                | Estimates local intensity changes using finite differences                                                                             |
| **Custom Convolution Kernel**         | `cv2.filter2D(image, -1, kernel)` or `scipy.ndimage.convolve()`                       | Designing custom edge detectors; research; specialized image domains                               | When a standard, well-tested detector is sufficient                               | Applies an arbitrary convolution kernel to detect specific spatial patterns                                                            |

<hr/>

### Symmetrical Filter (Prewitt) for Edge Detection

**Description**: 
> The Prewitt filter is used for edge detection in images by calculating the gradient of the image intensity at each pixel.

- implementation of Prewitt filter for edge detection in images using OpenCV and NumPy.

```python
import cv2
import numpy as np

img = np.ndarray([])

h_kernel = [[-1, -1, -1], # define horizontal kernel
            [0, 0, 0],
            [1, 1, 1]]

v_kernel = [[-1, 0, 1], # define vertical kernel
            [-1, 0, 1],
            [-1, 0, 1]]

img_hf = cv2.filter2D(img, ddepth=1, kernel=h_kernel) # apply horizontal filter
img_vf = cv2.filter2D(img, ddepth=1, kernel=v_kernel) # apply vertical filter
```

<hr/>

### Averaging Filter (Sobel Filter)
**Description**:
> The Sobel filter is used for edge detection in images by calculating the gradient of the image intensity at each pixel, similar to the Prewitt filter, but it gives more weight to the central pixels, making it more sensitive to edges.

- implementation of Sobel filter for edge detection in images using OpenCV and NumPy.

```python
import cv2
import numpy as np

img = np.ndarray([])

h_kernel = [[-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]] # define horizontal kernel

v_kernel = [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]] # define vertical kernel

img_hf = cv2.filter2D(img, ddepth=1, kernel=h_kernel) # apply horizontal filter
img_vf = cv2.filter2D(img, ddepth=1, kernel=v_kernel) # apply vertical filter
```

- `cv2.Sobel` - detects edges in an image using the Sobel operator.
- - cv2.Sobel(src, ddepth, dx=1, dy, ksize=3) - detect edges by horizontal and vertical gradients. The parameters are:
  - `src` - input image
  - `ddepth` - desired depth of the output image
  - `dx` - order of the derivative in x direction
  - `dy` - order of the derivative in y direction
  - `ksize` - size of the extended Sobel kernel; it must be 1, 3, 5, or 7.

<hr/>

### Canny Edge Detector

**Description**:
> The Canny edge detector is a multi-stage algorithm for detecting edges in images. It uses a combination of Gaussian filtering, gradient calculation, non-maximum suppression, double thresholding, and edge tracking by hysteresis to produce a binary image of edges.

- `cv2.Canny` - detects edges in an image using the Canny edge detection algorithm.

Algorithm Steps:

1. Noise Reduction (Gaussian Filter)
2. Gradient Calculation (Sobel Filter)
3. Non-maximum Suppression (NMS)
4. Double Thresholding (Hysteresis Thresholding) Th + Tl ( G> Th = edge, G < Tl = No Edge, Tl < G < Th = Potential Edge)
5. Edge Tracking by Hysteresis

<hr/>

### Hough Transformation
**Description**:
> The Hough Transform is a feature extraction technique used in image analysis, computer vision, and digital image processing. It is used to detect simple shapes such as lines, circles, and ellipses in images.

- `cv2.HoughLines` - detects lines in an image using the Hough Transform.
- `cv2.HoughCircles` - detects circles in an image using the Hough Transform.

<hr/>

### Homework Feedback

- Very good! In practice we don't process each video frame separately but we also use temporal information, i.e., the detections from previous frames. 
- By combining detections from several frames we can stabilize the output and obtain more accurate detections.