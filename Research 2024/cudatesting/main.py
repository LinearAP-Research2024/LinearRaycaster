import torch
import torchvision
"""
print(torch.cuda.is_available())
print("hello world!python")
print(torch.cuda.get_device_name(0))
"""
print("AP Research CUDA Tests: PiP comparisons: \n"
      "Is CUDA working: " + str(torch.cuda.is_available()) + "\n"
      "Device Name: " + str(torch.cuda.get_device_name(0)) + "\n"
    )
next(net.parameters()).is_cuda
