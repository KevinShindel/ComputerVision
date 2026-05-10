## Image Geometric Transformation

- Translation: - move image (shift in (x,y) only, no rotation/scale)
- Euclidean: - rotate and translate (keeps lengths and angles)
- Similarity: - rotate, translate, and uniform scale (keeps angles, same scale in all directions)
- Projectile: - perspective (homography) transform (can change parallel lines to meet; models camera viewpoint change)
- Affine: -  linear transform + translation (rotation/scale/shear; keeps parallel lines parallel)

<img style="border-radius:5px; width:50%; heigh: auto; border: white 1px" src="https://www.researchgate.net/publication/329973750/figure/fig1/AS:867677526564864@1583881851548/Different-types-of-geometric-transformations-on-an-input-image-original.png" alt="transformation matrix"/>

<img style="border-radius:5px; width:50%; heigh: auto; border: white 1px" src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*VbIO38pdDG52zVK8_zkFGA.png" alt="Image warping"/>

## Homogeneous Coordinates 
- Point representation using an extra dimension \(`x, y,1`\) that enables affine and projective transforms with matrix multiplication.

## Euclidean Coordinates 
- Standard2D/3D Cartesian coordinates \(`x, y` or `x, y, z`\) without the extra homogeneous component.

## Affine transformation 
- A linear transform plus translation that preserves parallel lines but not lengths or angles.

## Homography
- A projective transform represented by a `3x3` matrix that maps points between planes and

## RANSAC algorithm
- Robust statistical method for fitting models to data with outliers
- Iteratively selects random subsets of data points to fit a model
- Counts inliers (points fitting the model within a threshold) for each iteration
- Keeps the model with the maximum number of inliers
- Very slow


## Recommended Materials
1. [Robust Estimation : RANSAC](https://www.cse.psu.edu/~rtc12/CSE486/lecture15.pdf)
2. [Basic concepts of the homography explained with code](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)
3. [Multi-frame super-resolution](https://arxiv.org/pdf/1905.03277)
4. [Mesh-to-grid reconstruction](https://arxiv.org/pdf/2205.11202)

## Homework Feedback
Well done! Just keep in mind that in order to rectify a document you need to preserve its aspect ratio (which you did not).

To run an OCR (Optical Character Recognition) afterwards, we'd need a better image resolution.
BTW if you need to run OCR you may find this library helpful https://pypi.org/project/pytesseract/