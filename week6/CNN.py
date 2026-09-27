import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

MNIST.mirrors = ["https://storage.googleapis.com/cvdf-datasets/mnist/"]

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)   # 1→8 قنوات، يحافظ على 28x28
        self.pool1 = nn.MaxPool2d(2, 2)                          # 28x28 → 14x14
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 8→16 قناة، يحافظ على 14x14
        self.pool2 = nn.MaxPool2d(2, 2)                          # 14x14 → 7x7
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 7 * 7, 10)                      # مخرج نهائي: 10 أرقام

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Avg Loss: {total_loss/len(train_loader):.4f}")

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"دقة CNN على بيانات الاختبار: {100 * correct / total:.2f}%")