## Gradient Descent Algorithm

$$
x_{\text{new}} = \left\{x_{\text{old}} - \alpha \,\nabla f(x_{\text{old}})\right\}
$$

Where:
- $x_{\text{new}}$ is the updated value of the variable we are optimizing
- $x_{\text{old}}$ is the current value of the variable
- $\alpha$ is the learning rate, which controls the step size of the update
- $\nabla f(x_{\text{old}})$ is the gradient of the function $f$ at the current value of $x_{\text{old}}$, which indicates the direction of the steepest ascent. We subtract it to move in the direction of steepest descent, which helps us find the minimum of the function.


## Gradient Descent Implementation in Python
```python
drink_1 = 1
bill_1 = 5

drink_2 = 2
bill_2 = 10
learning_rate = 0.1

w = 4
grad = 2*(w*drink_1-6)*1 + 2*(w*2-10)*2
w = w - (grad) * learning_rate

```


```python
import numpy as np

def gradient_descent(grad_f, x0, learning_rate=0.01, max_iterations=1000):
    x = x0
    for i in range(max_iterations):
        gradient = grad_f(x)
        x = x - learning_rate * gradient
        if np.linalg.norm(gradient) < 1e-6:  # Convergence criterion
            print(f"Converged after {i+1} iterations.")
            break
    return x
    
# Example usage:
def f(x):
    return x**2 + 4*x + 4
    
def grad_f(x):
    return 2*x + 4  
    
initial_guess = 0.0
optimal_x = gradient_descent(f, grad_f, initial_guess)
print(f"Optimal x: {optimal_x}, f(x): {f(optimal_x)}")
```


## Recommendation Link 
- [Neural Networks and Deep Learning Complete Course](https://www.youtube.com/watch?v=E13qqHb3J7U)
- [Crash Course on Multi-Layer Perceptron Neural Networks](https://machinelearningmastery.com/neural-networks-crash-course/)
- [Optimizers Explained](https://towardsdatascience.com/neural-network-optimizers-from-scratch-in-python-af76ee087aab/)
- [Gradient Descent Algorithm](http://towardsdatascience.com/complete-step-by-step-gradient-descent-algorithm-from-scratch-acba013e8420/)