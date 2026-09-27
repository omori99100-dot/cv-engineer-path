import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# بيانات XOR: مدخلين، مخرج واحد
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])  # XOR: نفس القيم=0، مختلفة=1

np.random.seed(1)

# أوزان الطبقة المخفية (2 مدخل → 4 خلايا مخفية)
W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))

# أوزان طبقة المخرج (4 خلايا مخفية → 1 مخرج)
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

learning_rate = 0.5

for epoch in range(5000):
    # ===== Forward =====
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)              # مخرجات الطبقة المخفية

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)              # المخرج النهائي

    loss = np.mean((y - a2) ** 2)

    # ===== Backward =====
    d_loss_a2 = -2 * (y - a2)
    d_a2_z2 = sigmoid_derivative(a2)
    delta2 = d_loss_a2 * d_a2_z2                    # خطأ طبقة المخرج

    d_z2_a1 = W2.T
    delta1 = np.dot(delta2, d_z2_a1) * sigmoid_derivative(a1)  # خطأ الطبقة المخفية

    # تحديث الأوزان
    W2 -= learning_rate * np.dot(a1.T, delta2)
    b2 -= learning_rate * np.sum(delta2, axis=0, keepdims=True)
    W1 -= learning_rate * np.dot(X.T, delta1)
    b1 -= learning_rate * np.sum(delta1, axis=0, keepdims=True)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nالتنبؤات النهائية:")
print(a2)