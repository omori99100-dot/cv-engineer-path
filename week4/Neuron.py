import numpy as np

# مدخلات (مثلاً: 3 قياسات من قطعة صناعية — طول، وزن، سماكة)
X = np.array([0.5, 0.8, 0.2])

# أوزان عشوائية أولية (نفس عدد المدخلات)
weights = np.array([0.4, -0.6, 0.9])

# bias (قيمة إضافية ثابتة)
bias = 0.1

# الخطوة 1: الضرب والجمع (Weighted Sum)
z = np.dot(X, weights) + bias
print("z (weighted sum):", z)

# الخطوة 2: دالة التفعيل (Sigmoid) - تحول الناتج لقيمة بين 0 و1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

output = sigmoid(z)
print("output (after activation):", output)
# نفترض القيمة الصحيحة (الحقيقية) لهذا المثال معروفة: القطعة فعليًا معيبة (لنقل نمثلها بـ 1)
y_true = 1

# الناتج اللي حسبناه قبل شوي
y_pred = output  # كان 0.5

# Mean Squared Error - أبسط دالة خسارة
loss = (y_true - y_pred) ** 2
print("Loss:", loss)