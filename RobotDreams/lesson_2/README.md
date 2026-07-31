# Lesson 2 ( Pixel Operations)

This folder contains Lesson 2 materials focused on tone/intensity transformations and histogram-based image processing.

<br/>

## Gamma Correction Overview

| Method                                 | Short description                           | Implementation (Python/OpenCV)                                                                                                      |
|----------------------------------------|---------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Fixed global gamma                     | One (\gamma) for whole image.               | out = np.clip((img/255.0)gamma * 255, 0, 255).astype(np.uint8)                                                                      | 
| Manual gamma tuning                    | User picks best (\gamma).                   | for g in [0.6,0.8,1.0,1.2,1.4]: show((img/255.0)g)                                                                                  |
| Auto gamma from mean luminance         | Compute (\gamma) from image brightness.     | m = gray.mean()/255.0; gamma = np.log(target)/np.log(max(m,1e-6)); out=((img/255.0)gamma255).astype(np.uint8)                       |
| LUT-based gamma                        | Faster global gamma via lookup table.       | lut = np.array([((i/255.0)gamma)255 for i in range(256)], np.uint8); out = cv2.LUT(img, lut)                                        |
| Per-channel gamma (RGB/BGR)            | Different (\gamma) per channel.             | b,g,r = cv2.split(img); out = cv2.merge([f(b,gb), f(g,gg), f(r,gr)])                                                                |
| Luminance-only gamma                   | Apply gamma only to brightness channel.     | lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB); L,a,b=cv2.split(lab); L=f(L,gamma); out=cv2.cvtColor(cv2.merge([L,a,b]),cv2.COLOR_LAB2BGR) |
| Piecewise gamma                        | Different gamma for dark/mid/bright ranges. | mask1=img<85; mask2=(img>=85)&(img<170); ...; out[mask1]=f(img[mask1],g1)                                                           |
| Spatially adaptive gamma               | Gamma depends on local illumination.        | illum=cv2.GaussianBlur(gray,(0,0),15); gamma_map=1.8-illum/255.0; out=((img/255.0)gamma_map[...,None]255).astype(np.uint8)          |
| Weighted / brightness-preserving gamma | Brighten shadows, protect highlights.       | w = 1 - gray/255.0; gamma_map = 1 + alpha(w-0.5); out=((img/255.0)gamma_map[...,None]255).astype(np.uint8)                          |
| Multi-scale gamma fusion               | Blend outputs from several gammas.          | i1=f(img,0.7); i2=f(img,1.0); i3=f(img,1.4); out=cv2.addWeighted(i1,0.4,i2,0.4,0); out=cv2.addWeighted(out,1.0,i3,0.2,0)            |

## Contents

- `Homework.md`  
  Homework tasks for Lesson 2: practice applying intensity transformations and analyzing results (plots/figures, short conclusions).

- `Gamma.ipynb`  
  Notebook on gamma correction (non\‑linear luminance mapping): how the gamma parameter changes contrast, how it affects dark vs bright tones, and why gamma matters for perception and quantization.

- `Histogram.ipynb`  
  Notebook on image histograms: computing and visualizing intensity/channel histograms, interpreting exposure/contrast issues, and using histogram-based adjustments (e.g., stretching/equalization) to improve image appearance.

<br/>


## Requirements

- Python
- Packages: `opencv-python`, `numpy`, `matplotlib`

<br/>



## White Balance Algorithms Overview

| Алгоритм                  | Основна ідея                                   | Швидкість  | Стійкість до шуму                           | Типові випадки використання                                               |
|---------------------------|------------------------------------------------|------------|---------------------------------------------|---------------------------------------------------------------------------|
| **Scale-by-Max**          | Нормалізувати кожен канал за його максимумом   | ⭐⭐⭐⭐⭐ | ⭐                                          | Просте нормалізування, попередня обробка                                  |
| **Gray World**            | Середній колір сцени має бути нейтрально-сірим | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐                                    | Загальні природні сцени, класичний color constancy                        |
| **White Patch (Max-RGB)** | Найяскравіший об'єкт повинен бути білим        | ⭐⭐⭐⭐⭐ | ⭐⭐ (⭐⭐⭐⭐ при використанні перцентиля) | Сцени з білими еталонними об'єктами або яскравими відбиваючими поверхнями |


## Advanced White Balance Algorithms Overview

| Алгоритм           | Принцип                                                           | Швидкість  | Якість компенсації освітлення | Збереження кольорів | Типові застосування                                                                   |
|--------------------|-------------------------------------------------------------------|------------|-------------------------------|---------------------|---------------------------------------------------------------------------------------|
| **Shades of Gray** | Узагальнення Gray World через Minkowski p-норму                   | ⭐⭐⭐⭐   | ⭐⭐⭐⭐                      | ⭐⭐⭐⭐            | Загальні задачі Computer Vision                                                       |
| **Gray Edge**      | Використовує статистику градієнтів замість кольорів               | ⭐⭐⭐     | ⭐⭐⭐⭐⭐                    | ⭐⭐⭐⭐            | Робототехніка, автономне керування, сцени з домінуючим кольором                       |
| **SSR**            | Логарифмічне відношення між зображенням і його Gaussian-розмиттям | ⭐⭐⭐⭐   | ⭐⭐⭐                        | ⭐⭐                | Покращення локального контрасту                                                       |
| **MSR**            | Усереднення кількох SSR із різними масштабами                     | ⭐⭐⭐     | ⭐⭐⭐⭐⭐                    | ⭐⭐⭐              | Нерівномірне освітлення, тіні                                                         |
| **MSRCR**          | MSR + функція відновлення кольору (Color Restoration)             | ⭐⭐       | ⭐⭐⭐⭐⭐                    | ⭐⭐⭐⭐⭐          | Професійна обробка зображень, медичні та супутникові дані, системи комп'ютерного зору |


<br/>


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

<br/>


## Usefully Links

> - [OpenCV tutorial for HE](https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html)
> - [Detailed explanation of gamma correction](https://www.cambridgeincolour.com/tutorials/gamma-correction.htm)
> - [Histogram Equalization](https://onlinelibrary.wiley.com/doi/10.1155/2021/8883571)


## Homework Feedback
- Все верно! Как можно видеть, если на картинке есть какие то светлые пикселы, то скейлинг на максимум не имеет большого эффекта.