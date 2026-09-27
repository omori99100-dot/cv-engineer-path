import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# تأكد وين بايثون شايف نفسه
print(os.getcwd())

# غيّر الاسم لاسم صورتك الفعلي، وحطها بنفس فولدر السكربت
image_path = r"C:\Users\A Center\Downloads\Telegram Desktop\photo_30_2026-05-31_17-41-44.jpg"

img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"ما قدرت أفتح الصورة: {image_path} — تأكد المسار صحيح")

# OpenCV يقرأ بصيغة BGR، matplotlib يتوقع RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# تحويل لرمادي
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel: يحسب التدرج (gradient) بالاتجاه الأفقي والعمودي
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Canny: خوارزمية جاهزة تدمج Sobel + thresholding لتعطيك حواف نظيفة (خط أبيض على أسود)
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
axes2[0].imshow(gray, cmap='gray')
axes2[0].set_title("Gray")

axes2[1].imshow(sobel_x, cmap='gray')
axes2[1].set_title("Sobel X")

axes2[2].imshow(edges, cmap='gray')
axes2[2].set_title("Canny")

#plt.show()

print("shape original:", img_rgb.shape)   # (H, W, 3)
print("shape gray:", gray.shape)          # (H, W)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_rgb)
axes[0].set_title("Original")

axes[1].imshow(gray, cmap='gray')
axes[1].set_title("Grayscale")



plt.figure(figsize=(6,5))
plt.imshow(sobel_y, cmap='gray')
plt.title("Sobel Y فقط")
#plt.show()

# Threshold: نحول الصورة الرمادية لثنائية (أبيض/أسود)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Kernel: مصفوفة صغيرة (زي 5x5) تحدد حجم "الفرشاة" اللي تستخدمها Erosion/Dilation
kernel = np.ones((3,3), np.uint8)

eroded = cv2.erode(binary, kernel, iterations=1)
dilated = cv2.dilate(binary, kernel, iterations=1)

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
axes3[0].imshow(binary, cmap='gray')
axes3[0].set_title("Binary (Threshold)")

axes3[1].imshow(eroded, cmap='gray')
axes3[1].set_title("Eroded")

axes3[2].imshow(dilated, cmap='gray')
axes3[2].set_title("Dilated")

adaptive = cv2.adaptiveThreshold(
    gray, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 
    blockSize=11, C=2
)

cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)

cv2.imwrite("cleaned_document.jpg", cleaned)

fig4, axes4 = plt.subplots(1, 2, figsize=(10, 5))
axes4[0].imshow(adaptive, cmap='gray')
axes4[0].set_title("Adaptive Threshold")
axes4[1].imshow(cleaned, cmap='gray')
axes4[1].set_title("Cleaned (Closing)")

plt.show()
