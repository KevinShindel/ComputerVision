## CUDA Driver 
[NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

## PyTorch

### NVIDIA RTX 5060 Ti (Blackwell architecture)
The RTX 5060 Ti uses the Blackwell architecture (SM 12.0) and requires **CUDA 12.8+**.
Use the following command to install PyTorch with CUDA 12.8 support:

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Verify GPU is available:
```python
import torch
print(torch.cuda.is_available())      # True
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 5060 Ti
```

For other NVIDIA GPUs, see the [PyTorch CUDA Installation Page](https://pytorch.org/get-started/locally/)

## TensorFlow 

> Caution: TensorFlow 2.10 was the last TensorFlow release that supported GPU on native-Windows.
> Starting with TensorFlow 2.11, you will need to install TensorFlow in WSL2,
> or install tensorflow or tensorflow-cpu and, optionally, try the TensorFlow-DirectML-Plugin

[TensorFlow CUDA Installation Page](https://www.tensorflow.org/install/pip#windows-native)