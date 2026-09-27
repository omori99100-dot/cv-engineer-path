import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\A Center\Downloads\Telegram Desktop\photo_42_2026-05-31_17-41-44.jpg")  # صورة ملونة فيها ألوان مميزة (جرب صورة فيها لون واضح كأحمر/أزرق)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# مثال: فرز اللون الأحمر (نطاق H تقريبي)
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

mask = cv2.inRange(hsv, lower_red, upper_red)
result = cv2.bitwise_and(img, img, mask=mask)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Original")
plt.subplot(1, 3, 2); plt.imshow(mask, cmap='gray'); plt.title("Mask (Red Detection)")
plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); plt.title("Filtered Result")
plt.show()