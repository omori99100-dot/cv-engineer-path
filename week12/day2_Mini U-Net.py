import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # ===== Encoder (نازل) =====
        self.enc1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # ===== Bottleneck (أضيق نقطة، منتصف الـU) =====
        self.bottleneck = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # ===== Decoder (صاعد) =====
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        # طبقة نهائية: تحول لعدد الفئات المطلوب
        self.final = nn.Conv2d(16, 2, kernel_size=1)  # مثال: فئتين (خلفية/عيب)

    def forward(self, x):
        x1 = torch.relu(self.enc1(x))
        x = self.pool1(x1)
        x2 = torch.relu(self.enc2(x))
        x = self.pool2(x2)

        x = torch.relu(self.bottleneck(x))

        x = self.up2(x)
        x = torch.relu(self.dec2(x))
        x = self.up1(x)
        x = torch.relu(self.dec1(x))

        return self.final(x)


model = SimpleUNet()
sample = torch.randn(1, 3, 64, 64)
output = model(sample)
print("شكل المدخل:", sample.shape)
print("شكل المخرج:", output.shape)