import torch
import torch.nn as nn

# Convolutional layer: يتعلم عدة "فلاتر" تلقائيًا
conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)

# صورة تجريبية (batch=1, channel=1, height=28, width=28) — نفس شكل MNIST
sample = torch.randn(1, 1, 28, 28)
output = conv(sample)

print("شكل المدخل:", sample.shape)
print("شكل المخرج:", output.shape)

conv_padded = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
output_padded = conv_padded(sample)
print("شكل المخرج مع padding:", output_padded.shape)