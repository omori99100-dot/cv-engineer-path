import cv2
import numpy as np
import matplotlib.pyplot as plt

# قراءة الصورة
img = cv2.imread(r"C:\Users\A Center\Downloads\Telegram Desktop\photo_43_2026-05-31_17-41-44.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Kernel للـ blur: كل بكسل يصير متوسط جيرانه (15x15)
blur_kernel = np.ones((15, 15), np.float32) / 225
blurred = cv2.filter2D(gray, -1, blur_kernel)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
adaptive = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, blockSize=15, C=5
)


# نبدأ من صورة ثنائية (استخدم adaptive threshold من الأسبوع الماضي)
adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, blockSize=11, C=2
)

contours, hierarchy = cv2.findContours(
    adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

print("عدد الكائنات المكتشفة:", len(contours))

min_area = 50  # جرّب أرقام مختلفة وشوف الأثر
significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]

print("قبل الفلترة:", len(contours))
print("بعد الفلترة:", len(significant_contours))

img_filtered = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_filtered, significant_contours, -1, (0, 255, 0), 3)
plt.figure(figsize=(8, 10))
plt.imshow(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB))
plt.title(f"Filtered Contours: {len(significant_contours)}")
plt.show()

# رسم الكفافات على نسخة ملونة من الصورة
img_with_contours = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 3)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(gray, cmap='gray'); plt.title("Original")
plt.subplot(1, 3, 2); plt.imshow(adaptive, cmap='gray'); plt.title("Binary")
plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB)); plt.title(f"Contours: {len(contours)}")
plt.show()