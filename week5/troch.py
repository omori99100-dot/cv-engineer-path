import torch
import torch.nn as nn

X = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# تعريف الشبكة كـ class يرث من nn.Module
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4)   # نفس W1, b1 (2 مدخل → 4 خلايا)
        self.output = nn.Linear(4, 1)   # نفس W2, b2 (4 → 1 مخرج)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

model = XORNet()

loss_fn = nn.MSELoss()                                  # نفس MSE اللي حسبناه يدويًا
optimizer = torch.optim.SGD(model.parameters(), lr=0.5) # نفس Gradient Descent

for epoch in range(5000):
    optimizer.zero_grad()        # تصفير الـ gradients القديمة قبل كل خطوة
    y_pred = model(X)            # Forward
    loss = loss_fn(y_pred, y)    # حساب الخسارة
    loss.backward()              # Backward (نفس اليوم الماضي)
    optimizer.step()             # تحديث الأوزان (نفس weights -= lr * gradient يدويًا)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("\nالتنبؤات النهائية:")
print(model(X))