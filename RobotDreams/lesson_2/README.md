# Lesson 2 ( Pixel Operations)

This folder contains Lesson 2 materials focused on tone/intensity transformations and histogram-based image processing.

## Contents

- `Homework.md`  
  Homework tasks for Lesson 2: practice applying intensity transformations and analyzing results (plots/figures, short conclusions).

- `Gamma.ipynb`  
  Notebook on gamma correction (non\‑linear luminance mapping): how the gamma parameter changes contrast, how it affects dark vs bright tones, and why gamma matters for perception and quantization.

- `Histogram.ipynb`  
  Notebook on image histograms: computing and visualizing intensity/channel histograms, interpreting exposure/contrast issues, and using histogram-based adjustments (e.g., stretching/equalization) to improve image appearance.

## Requirements

- Python
- Packages: `opencv-python`, `numpy`, `matplotlib`


## White Balance Algorithms Overview

| Algorithm    | When to Use                                                                                  | How to Choose / Detect Suitability                                                                                           | When It’s Not a Good Idea                                                                                  |
|--------------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| Gray World   | General color correction when lighting is unknown and no strong color dominance is expected. | Scene contains a mix of colors that should average to neutral gray overall (many different hues, no single color dominates). | Scenes dominated by a single color (e.g., ocean, forest, stage lighting) where the average is not neutral. |
| White Patch  | Color correction when a true white or very bright neutral surface is present.                | There is at least one reliable white/neutral highlight that should map to pure white.                                        | No true white pixels, or highlights are colored (specular, neon, colored lights).                          |
| Scale-By-Max | Color normalization to stretch channel intensities for improved contrast balance.            | Channels are clipped differently and you want each channel to reach full range.                                              | Images with noise in dark regions or with saturated highlights; can exaggerate noise and color casts.      |

## Histogram and Equalization Overview

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('image.jpg',0)

equalized = cv2.equalizeHist(img)
hist, bins = np.histogram(equalized.ravel(), bins=256, range=(0,255))
cdf = np.cumsum(hist/np.sum(hist))
plt.subplot(231), plt.plot(255*cdf), plt.axis('square'), plt.grid(True), plt.xlabel('Input'), plt.ylabel('Output')
plt.subplot(232), plt.hist(equalized)
```

## For color images  Option 1: ( need split and merge all channels ) 

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('image.jpg')

red, green, blue = cv2.split(image)

red = cv2.equalizeHist(red)
blue = cv2.equalizeHist(blue)
green = cv2.equalizeHist(green)

plt.imshow(cv2.merge([red, green, blue])), plt.axis(False), plt.title('equalizeHist')
```

## For color images Option 2: Equalize only the luma channel in HSV color space

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('image.jpg')

# Equalize luma only
image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
hue, saturation, value = cv2.split(image_hsv)

value = cv2.equalizeHist(value)
out = cv2.cvtColor(cv2.merge([hue, saturation, value]), cv2.COLOR_HSV2RGB)

plt.imshow(out), plt.axis(False), plt.title('equalizeHist on Value channel')
```

## Contrast Limited Adaptive Histogram Equalization (CLAHE)

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

img = cv2.imread('image.jpg',0)
cl1 = clahe.apply(img)
plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Image'), plt.axis('off')
plt.subplot(122), plt.imshow(cl1, 'gray'), plt.title('CLAHE Image'), plt.axis('off')
```

## Usefull Links

> - [OpenCV tutorial for HE](https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html)
> - [Detailed explanation of gamma correction](https://www.cambridgeincolour.com/tutorials/gamma-correction.htm)
> - [Histogram Equalization](https://onlinelibrary.wiley.com/doi/10.1155/2021/8883571)


## Homework Feedback
- Все верно! Как можно видеть, если на картинке есть какие то светлые пикселы, то скейлинг на максимум не имеет большого эффекта.