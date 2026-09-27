# 1. تصحيح ميل بسيط (نفترض 5 درجات ميل، زاوية شائعة بتركيب كاميرا)
import cv2
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r"D:\DMIC\Tmp_0037.dat\0 (4).JPG"  # أو مسار صورتك
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"ما قدرت أفتح الصورة: {image_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

(h, w) = gray.shape
center = (w // 2, h // 2)

angle_correction = cv2.getRotationMatrix2D(center, angle=-5, scale=1.0)
straightened = cv2.warpAffine(gray, angle_correction, (w, h), borderValue=255)

# 2. تنظيف ضجيج بـ Gaussian Blur خفيف (يحافظ على الحواف قدر الإمكان)
denoised = cv2.GaussianBlur(straightened, (5, 5), 0)

# 3. تكبير حدة الصورة بعد التنظيف (البلور دايمًا يطفي التفاصيل شوي، نعوّضها)
sharpen_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])
final_sharp = cv2.filter2D(denoised, -1, sharpen_kernel)

# 4. قص منطقة اهتمام (ROI) — مثلاً ربع الصورة العلوي الأيسر، وتكبيرها
roi = final_sharp[0:h//2, 0:w//2]
roi_zoomed = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# عرض البايبلاين كامل
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
axes[0].imshow(gray, cmap='gray'); axes[0].set_title("1. Original")
axes[1].imshow(straightened, cmap='gray'); axes[1].set_title("2. Straightened")
axes[2].imshow(denoised, cmap='gray'); axes[2].set_title("3. Denoised")
axes[3].imshow(final_sharp, cmap='gray'); axes[3].set_title("4. Sharpened")
axes[4].imshow(roi_zoomed, cmap='gray'); axes[4].set_title("5. ROI Zoomed")
plt.show()

cv2.imwrite("week2_pipeline_result.jpg", roi_zoomed)