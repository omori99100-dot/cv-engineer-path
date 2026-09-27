import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import DataLoader
import copy

# تحميل بيانات حقيقية (نمل/نحل) — تلقائيًا أول مرة
download_and_extract_archive(
    'https://download.pytorch.org/tutorial/hymenoptera_data.zip',
    download_root='./data'
)

# نفس الـ mean/std اللي استُخدمت وقت تدريب ResNet على ImageNet
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

train_data = datasets.ImageFolder('./data/hymenoptera_data/train', transform=transform)
val_data = datasets.ImageFolder('./data/hymenoptera_data/val', transform=transform)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

print("عدد فئات المهمة:", train_data.classes)
print("عدد صور التدريب:", len(train_data))
print("عدد صور التحقق:", len(val_data))


def build_model(finetune=False):
    model = models.resnet18(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)

    if finetune:
        for param in model.layer4.parameters():
            param.requires_grad = True

    return model


def train_model(model, optimizer, epochs=5):
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        print(f"Epoch {epoch}: Val Accuracy = {100*correct/total:.2f}%")


print("\n=== Transfer Learning البسيط (fc فقط) ===")
model_simple = build_model(finetune=False)
optimizer_simple = torch.optim.Adam(model_simple.fc.parameters(), lr=0.001)
train_model(model_simple, optimizer_simple)

print("\n=== Fine-tuning (layer4 + fc) ===")
model_finetune = build_model(finetune=True)
optimizer_finetune = torch.optim.Adam([
    {'params': model_finetune.layer4.parameters(), 'lr': 0.0001},
    {'params': model_finetune.fc.parameters(), 'lr': 0.001}
])
train_model(model_finetune, optimizer_finetune)