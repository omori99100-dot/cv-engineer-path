import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # مشتقة Sigmoid لها صيغة بسيطة جدًا بدلالة نفس ناتج Sigmoid
    return x * (1 - x)

X = np.array([0.5, 0.8, 0.2])
weights = np.array([0.4, -0.6, 0.9])
bias = 0.1
y_true = 1  # القطعة فعليًا معيبة

learning_rate = 0.5

for step in range(5):
    # Forward
    z = np.dot(X, weights) + bias
    y_pred = sigmoid(z)
    loss = (y_true - y_pred) ** 2

    # Backward (حساب Gradient يدويًا بقاعدة السلسلة Chain Rule)
    d_loss_d_pred = -2 * (y_true - y_pred)
    d_pred_d_z = sigmoid_derivative(y_pred)
    d_z_d_weights = X

    gradient = d_loss_d_pred * d_pred_d_z * d_z_d_weights

    # تحديث الأوزان بعكس اتجاه الـ Gradient
    weights = weights - learning_rate * gradient

    print(f"Step {step}: loss={loss:.4f}, y_pred={y_pred:.4f}")