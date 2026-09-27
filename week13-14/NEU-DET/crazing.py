from ultralytics import YOLO

# مسار النموذج بعد ما نزّلته من Colab لجهازك
model = YOLO(r"C:\Users\A Center\cv_engineer_path\week14\yolov8n.pt")

image_paths = [
    r"C:\Users\A Center\Downloads\images\01_metal_scratch.png",
    r"C:\Users\A Center\Downloads\images\02_plastic_crack.png",
    r"C:\Users\A Center\Downloads\images\03_coating_peel.png",
    r"C:\Users\A Center\Downloads\images\04_surface_dent.png",
]

results = model(image_paths, conf=0.25)

for i, result in enumerate(results):
    print(f"\n=== صورة {i+1}: {image_paths[i].split(chr(92))[-1]} ===")
    if len(result.boxes) == 0:
        print("ما اكتشف أي عيب")
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"الفئة: {model.names[cls]}, الثقة: {conf:.2f}")
    result.show()