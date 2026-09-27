import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\A Center\Downloads\Telegram Desktop\photo_42_2026-05-31_17-41-44.jpg")  # صورة ملونة فيها ألوان مميزة (جرب صورة فيها لون واضح كأحمر/أزرق)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# رسم الـ histogram قبل أي تعديل
plt.figure(figsize=(6, 4))
plt.hist(gray.ravel(), bins=256, range=[0, 256])
plt.title("Histogram - Original")
plt.xlabel("Pixel Value")
plt.ylabel("Count")
plt.show()

equalized = cv2.equalizeHist(gray)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); plt.imshow(gray, cmap='gray'); plt.title("Original")
plt.subplot(1, 2, 2); plt.imshow(equalized, cmap='gray'); plt.title("Equalized")
plt.show()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_result = clahe.apply(gray)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(gray, cmap='gray'); plt.title("Original")
plt.subplot(1, 3, 2); plt.imshow(equalized, cmap='gray'); plt.title("Global Equalize")
plt.subplot(1, 3, 3); plt.imshow(clahe_result, cmap='gray'); plt.title("CLAHE")
plt.show()