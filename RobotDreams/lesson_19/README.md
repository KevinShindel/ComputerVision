# Patience
- Do -> Reflect -> Learn -> Repeat ( iterative process)

# Neural Networks Fail Silently
- Absences of errors != DNN works well

# ML learning setup
1. Learning / Evaluation Pipeline
2. loss function
3. metric monitoring
4. Always use fixed seed

# Dumb Baselines
- First use simple solution
- Understand good\bad performance

# Overfit
- Train model on single batch - If there are no overfitting - model is weak / logic bugs
- Safe Optimization - Adam

# Regularization
- More Data
- Augmentation
- Smaller input as can
- Less complex model
- Dropout
- Weight Decay
- Early Stopping
- **Important: Add regularization one-by-one !**

# Model Tuning
- Hyper-parameter tuning
- - Batch-size
- - Learning Rate
- - Optimizer
- - Other params
- Optuna ( HyperOpt )

# Squeeze the juice
- Use ensembles
- Train for long time

# Model Zoo
- Code & Weights for popular architectures

# Optuna Tune
- [MedianPruner (using the median stopping rule)](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html) 
- Use Optuna Visualization for understand results
- - optuna.visualization.plot_param_importances(study)
- - optuna.visualization.plot_parallel_coordinate(study)
- - optuna.visualization.plot_optimization_history(study)

# YOLO Tuning
- [Hyper-Params Tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/)

## Recommended Sources

1. [NN Fundamentals by Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
2. [Vision Transformers](https://theaisummer.com/vision-transformer/)
3. [Lane detection via self-attention](https://cardwing.github.io/files/DeepSAD.pdf)
4. [Large scale face dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)