import torch
import torch.nn as nn
from torchvision import models

# تحميل ResNet18 مُدرَّبة مسبقًا على ImageNet (مليون+ صورة، 1000 فئة)
model = models.resnet18(weights='IMAGENET1K_V1')

# تجميد كل الأوزان (نمنع تحديثها أثناء التدريب)
for param in model.parameters():
    param.requires_grad = False

# استبدال الطبقة الأخيرة بس (المُخصَّصة لمهمتنا الجديدة)
# لنفترض عندنا مهمة تصنيف ثنائية: "سليم" أو "معيب"
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

print(model.fc)
print("عدد البارامترات القابلة للتدريب:", sum(p.numel() for p in model.parameters() if p.requires_grad))