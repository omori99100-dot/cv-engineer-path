import ultralytics
# استخدام YOLO جاهز (نسخة مُدرَّبة مسبقًا) عبر مكتبة ultralytics
from ultralytics import YOLO

# تحميل نموذج YOLOv8 صغير، مُدرَّب مسبقًا على COCO (80 فئة شائعة)
model = YOLO('yolov8n.pt')

# تشغيل الكشف على صورة (أي صورة عندك فيها أشياء يومية: سيارة، شخص، كوب...)
results = model("D:\\DMIC\\Islamic Wallpapers\\753605771.jpg")

# عرض النتيجة
results[0].show()

# طباعة تفاصيل كل كائن مكتشف
for box in results[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    xyxy = box.xyxy[0].tolist()
    print(f"الفئة: {model.names[cls]}, الثقة: {conf:.2f}, الصندوق: {xyxy}")