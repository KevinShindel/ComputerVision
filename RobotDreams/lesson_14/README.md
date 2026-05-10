# Flatten Problem
- Do not store coords relations ( cause requence is flat )
- Far back for corners

## Convolution
- Conv2D layer
- - Filters - 16 - num of filters
- - KernelSize - (3x 3) - ( window size for filter )
- - Strides - (1, 1) step of convolution filter by vertical / horizontal (specifying the stride length of the convolution. strides > 1 is incompatible with dilation_rate > 1.)
- - padding - same, valid ("valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input. When padding="same" and strides=1, the output has the same size as the input.)
- - activation - ReLU/SeLU/LeakedReLU
- - Quantity of params per layer Conv2D = num_filters * kernel + bias = 16 + 3x3 + 1 = 160 params  

# Conv2D Logic (Padding/Kernel calculation)
padding = valid
formula =  floor((input_size - kernel_size) / strides) + 1

padding = same 
formula = ceil(input_size / strides)

# MaxPooling - Output Calculation
- pool_size = strides =  floor(input_size / strides)
- pool_size != strides = floor(input_size - pool_size ) / strides + bias

# Pooling Layers 

| Шар                  | Що зберігає         | Коли використовувати | Тип Моделі |
|----------------------|---------------------|----------------------|------------|
| Flatten              | Все + позиції       | складні залежності   | Маленька   |
| GlobalAveragePooling | середню присутність | класифікація         | Велика     |
| GlobalMaxPooling     | найсильніший сигнал | пошук ключових фіч   | Велика     |


## Kernel Optimal Size
- For large objects use large kernel size
- For zoomed objects use large kernel size
- For different scale size - use Pooling 

## Receptive Field
- Quantity of pixels that have importance 28 / 3 + 3 = 9 pixels
- Kernel size increasing 
- Receptive field — це область вхідного зображення, яка впливає на один нейрон.

## Pooling ( Decrease matrix algorithm )
- MaxPool config
- - Kernel Size 2x2
- - Stride 2x2
- AveragePooling make sense ( by task )

## CrossEntropy Loss Function Explained
- OneHotEncoding vs ProbabilityMatrix ( CNN Output )
- Correct classification function
- Strong Classification


## SoftMax Explained
- x = np.array([1.3, 7, 0.2, -2.4, 1])
- softmax(x) -> np.sum(x) = 1 -> 
- Probability Density Function

## Sparse vs CategoricalCrossEntropy ?


## Recommended Sources
- [Intro to pooling](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)
- [Cross-entropy](https://medium.com/data-science/cross-entropy-loss-function-f38c4ec8643e)
- [Softmax explained](https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d)
- [One-hot encoding](https://www.educative.io/blog/one-hot-encoding)