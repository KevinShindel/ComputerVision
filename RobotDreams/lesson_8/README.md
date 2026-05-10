# TODO: Find more usefull info for this file


# Search minimum for data

1. BruteForce - find global minimum
2. Optimizers - near global minimum

# Otsu Algorithm
- Search threshold for minimize dispersion
- Bimodal distribution (foreground/background)



### Recommended Resources

- [K-Means Clustering Algorithm](https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/)
- [OTSU](https://engineering.purdue.edu/kak/computervision/ECE661.08/OTSU_paper.pdf)
- [PATTERN RECOGNITION AND MACHINE LEARNING](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- [Image reconstruction by ML](https://arxiv.org/pdf/2205.11226)

### Homework Feedback

Very good and smart to verify your custom implementation against OpenCV's built-in Otsu as a sanity check ;-)
The problem with global thresholding is that it does not account for local changes of illumination (remember the chessboard optical illusion?). 
There are other more advanced and more adaptive thresholding methods like e.g.
https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_niblack_sauvola.html