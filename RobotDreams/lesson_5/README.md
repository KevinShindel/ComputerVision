# Lesson 5: DCT and Image Quantization

> This lesson covers frequency-domain representation with the Discrete Cosine Transform (DCT) and practical image quantization techniques used in compression.


## Famous Dithering Algorithms:

1. [Floyd-Steinberg Dithering](https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering)
2. [Atkinson dithering](https://en.wikipedia.org/wiki/Atkinson_dithering)

## Problem

- Raw data for Full HD image 1080x1920 = 2.0736 MPx x 3 channels (RGB) = 6.22 Mb
- Full HD video @ 25 FPS = 1 sec = 155.5 MB
- Full HD video @ 25 FPS = min = 9.33 GB

# Quantification

- 8bit x channel = 256x256x256 = 16.77 Billion colors
- Decrease number of colors with loss ( lossy compression )

# Dither

- is intentionally applied noise to (image) signal
- Goal: randomize quantization error
- poor visual quality but more perception
- If one pixel is dark ( lighter ) then neighbors makes lighter ( darker )

# Sparsity
- Modern standards work on sparsity ( jpeg, h.264, h.265)
- It is sparse signal that closer to zero
- Pixels that are closer to zero transformed as zero
- Lossy compression

# Discrete Cosinus Transformation ( aka DCT )
- Baseline functions ( cosinus waves w different frequencies)
- [Link on wiki](https://en.wikipedia.org/wiki/Discrete_cosine_transform)

# Peak Signal To Noise Ration ( aka PSNR )

- determine level noises on compression
- Using MSE
- If PSNR > 30 dB this good quality
- 

# Video codings

- [H.265](https://www.wowza.com/blog/hevc)
- [H.264](https://en.wikipedia.org/wiki/Advanced_Video_Coding)

# Techniques
- Quantization: reducing the number of bits needed to represent pixel values, often by mapping a range of values to a single representative value.
- DCT: transforming spatial pixel data into frequency components, allowing for efficient representation and compression by focusing on the most significant frequencies.
- Dithering: adding noise to quantized images to reduce visual artifacts and improve perceptual quality.
- Sparsity: leveraging the fact that many DCT coefficients are close to zero, enabling efficient compression by retaining only the most significant coefficients.

# 2D transformation
- Interpretation
- Cartesian product of 1D DCTs
- Conversion

# Noise detection
- DCT coefficients and noise
- PSNR and MSE metrics for quantization quality
- Zonal Sampling: retaining only low-frequency DCT coefficients to achieve compression while maintaining visual quality.

# [Image quality metrics](https://videoprocessing.ai/metrics/ways-of-cheating-on-popular-objective-metrics.html)
- PSNR - most frequent ( is not a good metric for quality estimation )
- SSIM - is almost insensitive to changes in brightness, contrast, hue and saturation
- MS-SIM - more accurate

# Metrics Comparison Table

![Metrics Comparsion](https://storage.videoprocessing.ai/benchmarks/metrics/psnr_and_ssim/pic25.png)

> Для моніторингу image quality metrics (PSNR, SSIM) під час тренування в TensorFlow використовуйте вбудовані 
> функції tf.image.psnr та tf.image.ssim як custom metrics у моделі та custom callback для логування.


```python
import tensorflow as tf

def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)  # Для нормалізованих [0,1]

def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

# У моделі:
model.compile(optimizer='adam',
              loss='mse',
              metrics=[psnr_metric, ssim_metric])  # Автоматичний моніторинг на epoch

```

## Notebooks

- `DCT.ipynb` — Introduction to the Discrete Cosine Transform for images: computing and visualizing DCT coefficients and understanding energy compaction.
- `ImageQuantization.ipynb` — Image quantization and dithering: comparing the original vs. quantized results and improving perceptual quality with Floyd–Steinberg error-diffusion dithering.

## Data

Sample images used in the notebooks are in `data/`.

## CV2 ML

- [KNN](https://docs.opencv.org/4.x/d8/d4b/tutorial_py_knn_opencv.html)
- [KMeans](https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html)
- [SVM](https://docs.opencv.org/4.x/dd/d3b/tutorial_py_svm_opencv.html)


## Homework Feedback
- Excellent, I like numba ;-)

Just a small detail here:

Keep values in valid range (prevents runaway/black artifacts)
> img_tmp[r, c + 1, :] = np.clip(img_tmp[r, c + 1, :], 0.0, 255.0)

Here you shouldn't clip the diffused image. Mathematically, if the diffusion causes the neighbour pixel to go beyond 255,
this just put more pressure on making the subsequent pixels darker to compensate.
We should just clip the final result.
There are other types of dithering, e.g. https://en.wikipedia.org/wiki/Atkinson_dithering , 
but Floyd-Steinberg is the most popular.