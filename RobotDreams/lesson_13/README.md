## Visualization
- Patterns
- Class balancing
- Data Distribution
- Draw Ground Truth with labels ( Random Image Sampling )

## Famous Datasets
- [FMNIST](https://en.wikipedia.org/wiki/Fashion-MNIST)
- [MNIST](https://en.wikipedia.org/wiki/MNIST_database)
- [GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html)

## Image Normalization
> Helps to decrease contrast and lightning
> 
x = x - np.mean(x) / np.std(x) ← Only for training


## Recommended Resources

- [Data normalization](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/)
- [Tensorflow built-in datasets](https://www.tensorflow.org/api_docs/python/tf/keras/datasets)
- [F1-Score](https://en.wikipedia.org/wiki/F-score)

## Homework Feedback

>Very good! The dataset is indeed highly **imbalanced**. <br />
> Moreover, there are classes that are on average almost **3 times brighter** than others. <br /> 
> A good data **normalization** (e.g. **standardization**) and balance compensation techniques (e.g. class weighting) should help here.