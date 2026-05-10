# Lesson 3 ( Linear Filtration)

This folder contains both the lesson materials and practice tasks:

- `lesson_3/ImageFiltering.ipynb` \- a hands\-on notebook covering image convolution and linear filtering in Python/OpenCV, including custom kernels, Gaussian blur, bilateral filtering, and denoising examples (Gaussian vs. median for salt\-and\-pepper noise).
- `lesson_3/Homework.md` \- homework tasks to apply the discussed filtering techniques in practice (smoothing, noise reduction, sharpening/unsharp masking, and edge\-preserving filters).

- Image Convolution - Applying a filter kernel to an image
- LowPass Filter - Good for noise reduction ( mean, median )
- HighPass Filter - Good for edge detection ( Sobel, Laplacian )
- Gaussian Filter ( blurring ) - Good for noise reduction ( camera noise )
- Bilateral Filter ( Spatial + Range ) - Good for edge preservation ( road sign detection )
- MedianBlur - Good for salt-and-pepper noise reduction ( impulse noise )
- Unsharp Masking - Good for image sharpening ( enhancing details )
- Guided Filter - Good for edge-preserving smoothing ( detail enhancement )



# Usefull links
> - [Intuitive Guide to Convolution](https://betterexplained.com/articles/intuitive-convolution/)
> - [OpenCV with Python Blueprints](https://subscription.packtpub.com/book/application-development/9781785282690/1)
> - [Bilateral Filtering for Gray and Color Images](https://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf)
> - [Guided Filter](https://en.wikipedia.org/wiki/Guided_filter)


## Homework Feedback
- BTW float32 would work too ;-)