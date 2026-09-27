from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# جرب على صورة ثانية فيها أكثر من كائن متشابه ومتقارب (لو عندك، أو صورة الأصص نفسها)
results = model("D:\\DMIC\\Islamic Wallpapers\\753605771.jpg", iou=0.5, conf=0.25)

print("عدد الصناديق النهائية بعد NMS:", len(results[0].boxes))

for box in results[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    print(f"{model.names[cls]}: {conf:.2f}")

results[0].show()