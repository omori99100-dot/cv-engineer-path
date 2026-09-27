import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

train_data = datasets.ImageFolder(
    r"C:\Users\A Center\cv_engineer_path\week15\train_organized",
    transform=transform
)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

print("الفئات:", train_data.classes)
print("عدد صور التدريب:", len(train_data))

model = models.resnet18(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)

optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("الجهاز المستخدم:", device)
model = model.to(device)

for epoch in range(3):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")

# حفظ النموذج المدرَّب لاستخدامه لاحقًا (خطوة جديدة، نحتاجها لاحقًا لتوليد التنبؤات)
torch.save(model.state_dict(), r"C:\Users\A Center\cv_engineer_path\week15\cats_dogs_model.pth")
print("تم حفظ النموذج بنجاح")