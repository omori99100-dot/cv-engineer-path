import torch
import torch.nn as nn

pool = nn.MaxPool2d(kernel_size=2, stride=2)

sample = torch.randn(1, 4, 28, 28)  # لنفترض 4 feature maps بحجم 28x28
output = pool(sample)

print("قبل Pooling:", sample.shape)
print("بعد Pooling:", output.shape)