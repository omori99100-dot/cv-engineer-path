import torch
import torch.nn as nn

class UNetWithSkip(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # لاحظ: 32 (من up2) + 32 (من enc2 عبر skip) = 64 مدخل هنا
        self.dec2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        # لاحظ: 16 (من up1) + 16 (من enc1 عبر skip) = 32 مدخل هنا
        self.dec1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.final = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, x):
        x1 = torch.relu(self.enc1(x))       # نحتفظ فيها للـ skip
        x = self.pool1(x1)
        x2 = torch.relu(self.enc2(x))       # نحتفظ فيها للـ skip
        x = self.pool2(x2)

        x = torch.relu(self.bottleneck(x))

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)       # الـ skip connection: نلصق x2 مباشرة
        x = torch.relu(self.dec2(x))

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)       # skip connection ثانية مع x1
        x = torch.relu(self.dec1(x))

        return self.final(x)


model = UNetWithSkip()
sample = torch.randn(1, 3, 64, 64)
output = model(sample)
print("شكل المخرج:", output.shape)