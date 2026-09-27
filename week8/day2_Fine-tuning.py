import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights='IMAGENET1K_V1')

# تجميد كل شي أول
for param in model.parameters():
    param.requires_grad = False

# استبدال الطبقة الأخيرة
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# فك تجميد آخر بلوك بس (layer4) + الطبقة الأخيرة الجديدة
for param in model.layer4.parameters():
    param.requires_grad = True

# Learning rate مختلف لكل مجموعة: صغير جدًا للطبقات المحررة القديمة، أكبر للطبقة الجديدة
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 0.0001},
    {'params': model.fc.parameters(), 'lr': 0.001}
])

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("عدد البارامترات القابلة للتدريب الآن:", trainable)