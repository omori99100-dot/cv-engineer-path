from ultralytics import YOLO

# نبدأ من نموذج مُدرَّب مسبقًا على COCO (نفس مبدأ Transfer Learning، الأسبوع 8-9)
model = YOLO('yolov8n.pt')

results = model.train(
    data=r"C:\Users\A Center\cv_engineer_path\week14\NEU-DET\data.yaml",
    epochs=30,
    imgsz=200,       # نفس أبعاد صور NEU-DET الأصلية (200×200)
    batch=16,
    name='neu_det_defects'
)
print ("تم تدريب نموذج YOLOv8 بنجاح على مجموعة بيانات NEU-DET")