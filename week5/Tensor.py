import torch

# نفس بيانات XOR بالضبط، لكن كـ PyTorch Tensor بدل NumPy array
X = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# نفس الأوزان، لكن requires_grad=True يخلي PyTorch "يراقب" هالمتغير لحساب Gradient لاحقًا
W1 = torch.randn(2, 4, requires_grad=True)
b1 = torch.zeros(1, 4, requires_grad=True)
W2 = torch.randn(4, 1, requires_grad=True)
b2 = torch.zeros(1, 1, requires_grad=True)

# Forward - لاحظ الشبه الكبير بكود NumPy الأسبوع الماضي
z1 = X @ W1 + b1              # @ تعني نفس np.dot
a1 = torch.sigmoid(z1)
z2 = a1 @ W2 + b2
a2 = torch.sigmoid(z2)

loss = torch.mean((y - a2) ** 2)
print("Loss:", loss.item())

# السحر هنا: بدل ما نحسب كل الـ deltas يدويًا زي الأسبوع الماضي، سطر واحد بس:
loss.backward()

print("Gradient لـ W1:")
print(W1.grad)