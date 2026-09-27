import cv2
import numpy as np
import matplotlib.pyplot as plt

# قراءة الصورة
img = cv2.imread(r"C:\Users\A Center\Downloads\Telegram Desktop\photo_43_2026-05-31_17-41-44.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Kernel للـ blur: كل بكسل يصير متوسط جيرانه (15x15)
blur_kernel = np.ones((15, 15), np.float32) / 225
blurred = cv2.filter2D(gray, -1, blur_kernel)

# مقارنة مع دالة GaussianBlur الجاهزة بـ OpenCV (أكثر استخدامًا عمليًا)
gaussian = cv2.GaussianBlur(gray, (15,15), 0)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(gray, cmap='gray'); plt.title("Original")
plt.subplot(1, 3, 2); plt.imshow(blurred, cmap='gray'); plt.title("Manual Blur")
plt.subplot(1, 3, 3); plt.imshow(gaussian, cmap='gray'); plt.title("Gaussian Blur")
plt.show()

gaussian_small = cv2.GaussianBlur(gray, (5, 5), 0)

gaussian_large = cv2.GaussianBlur(gray, (25, 25), 0)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(gaussian_small, cmap='gray'); plt.title("Kernel 5x5")
plt.subplot(1, 2, 2); plt.imshow(gaussian_large, cmap='gray'); plt.title("Kernel 25x25")
plt.show()

sharpen_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])
sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(gray, cmap='gray'); plt.title("Original")
plt.subplot(1, 2, 2); plt.imshow(sharpened, cmap='gray'); plt.title("Sharpened")
plt.show()


sharpen_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])
sharpened_strong = cv2.filter2D(gray, -1, sharpen_strong)
sharpened = cv2.filter2D(gray, -1, sharpen_kernel)


plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(gray, cmap='gray'); plt.title("Original")
plt.subplot(1, 3, 2); plt.imshow(sharpened, cmap='gray'); plt.title("Sharpen (center=5)")
plt.subplot(1, 3, 3); plt.imshow(sharpened_strong, cmap='gray'); plt.title("Sharpen (center=9)")
plt.show()