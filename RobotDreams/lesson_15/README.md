## Overfitting
- Train Dataset - visible for model ( for training purpose )
- Validation Dataset ( for hyperparameter optimization)
- Test Dataset ( for final estimation only )
- Unpredicted accuracy at production

## Loss vs Metric disbalance
- Difference between calculation

## Underfitting
- Worst for training / validation score
- Expected low accuracy at production

## Overfitting decreasing
- data addition 
- data augmentation
- Decrease model complexity ( use less layers )
- Use EarlyStopping on complex models

## Data Augmentation Recommendation
- Make sense only valid augmentations ( depends on production purpose )

## Dropout Regularization
- cut-off links between neurons for decrease overfitting

## Layer Regularization
- L2 Regularization ( add error for loss function )
- L1 Regularization
- Weight Decay

## Third Class for Binary Classification
- Add random images for third class ( mark them as unknown )
- Add descriptors
- TODO: ask this on QA session !

## Best Resolution
- Use the smallest resolution as can

## Recommended Resources
- [Overfitting vs Underfitting](https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/)
- [Train - validation - test split](https://artificial-intelligence-wiki.com/ai-tutorials/training-machine-learning-models/train-test-validation-split-methods/?utm_source=chatgpt.com)
- [Batch normalization](https://www.analyticsvidhya.com/blog/2021/03/introduction-to-batch-normalization/)
- [Fashion Mnist Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Albumentations Transforms](https://explore.albumentations.ai)