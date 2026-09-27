import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image_path = r"D:\DMIC\Tmp_0037.dat\0 (4).JPG"  # أو مسار صورتك
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"ما قدرت أفتح الصورة: {image_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

(h, w) = gray.shape
center = (w // 2, h // 2)

# مصفوفة التدوير: نقطة المركز، زاوية الدوران، عامل التكبير
rotation_matrix = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
rotated = cv2.warpAffine(gray, rotation_matrix, (w, h))
rotated_white_bg = cv2.warpAffine(gray, rotation_matrix, (w, h), borderValue=255)

scaled_up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
scaled_down = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)


plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(gray, cmap='gray'); plt.title("Original")
plt.subplot(1, 3, 2); plt.imshow(scaled_down, cmap='gray'); plt.title("Scaled 0.5x")
plt.subplot(1, 3, 3); plt.imshow(rotated_white_bg, cmap='gray'); plt.title("Rotated 45°")
plt.show()