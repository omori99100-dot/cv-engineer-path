import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# نغيّر مصدر التحميل لمرآة بديلة أكثر استقرارًا
MNIST.mirrors = ["https://storage.googleapis.com/cvdf-datasets/mnist/"]

# باقي الكود زي ما هو
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# تحميل بيانات MNIST (يتحمّل تلقائيًا أول مرة)
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print("عدد صور التدريب:", len(train_data))
print("شكل صورة واحدة:", train_data[0][0].shape)  # [1, 28, 28]

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()               # يحوّل الصورة 28x28 إلى متجه واحد 784
        self.fc1 = nn.Linear(28*28, 128)           # 784 مدخل → 128 خلية مخفية
        self.fc2 = nn.Linear(128, 10)              # 128 → 10 مخرجات (رقم 0 إلى 9)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))                # ReLU بدل Sigmoid هنا
        x = self.fc2(x)
        return x

model = MNISTNet()
loss_fn = nn.CrossEntropyLoss()                    # دالة خسارة مختلفة عن MSE
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

# تقييم الدقة على بيانات الاختبار
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"الدقة على بيانات الاختبار: {100 * correct / total:.2f}%")