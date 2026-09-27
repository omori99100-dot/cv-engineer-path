import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# نموذج Semantic Segmentation مُدرَّب مسبقًا (DeepLabV3، مُدرَّب على COCO)
model = deeplabv3_resnet50(weights='DEFAULT')
model.eval()

img = Image.open(r"D:\Telegram Desktop\photo_1_2025-05-21_00-46-25.jpg").convert('RGB')
transform = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)['out'][0]

# كل بكسل يحصل على أعلى فئة محتملة (نفس argmax اللي أخذناه بالأسبوع الخامس)
segmentation_mask = output.argmax(0).numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(segmentation_mask, cmap='tab20')
plt.title("Semantic Segmentation Mask")
plt.show()

