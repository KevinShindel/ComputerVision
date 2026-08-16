# Windows requirements

## Minimum validated setup

- Windows 10/11 x64
- NVIDIA driver `610.88+`
- Python `3.14.7+`
- NVIDIA GeForce RTX 5060 Ti

## PyTorch GPU install

Use the CUDA 12.8 wheels for the RTX 5060 Ti:

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

## Notes

- The CUDA toolkit is not required for standard Ultralytics training when using the PyTorch CUDA wheels.
- Verify GPU access with `nvidia-smi` and `torch.cuda.is_available()`.
