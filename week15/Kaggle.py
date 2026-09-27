import os
import shutil
from torchvision import datasets

train_dir = r"C:\Users\A Center\cv_engineer_path\week15\train"
train_data = datasets.ImageFolder(r"C:\Users\A Center\cv_engineer_path\week15\train_organized", transform=None)
organized_dir = r"C:\Users\A Center\cv_engineer_path\week15\train_organized"

# إنشاء مجلدين فرعيين
os.makedirs(os.path.join(organized_dir, "cat"), exist_ok=True)
os.makedirs(os.path.join(organized_dir, "dog"), exist_ok=True)

files = [f for f in os.listdir(train_dir) if f.endswith('.jpg')]
print(f"عدد الصور الكلي: {len(files)}")

for filename in files:
    if filename.startswith('cat'):
        shutil.copy(
            os.path.join(train_dir, filename),
            os.path.join(organized_dir, "cat", filename)
        )
    elif filename.startswith('dog'):
        shutil.copy(
            os.path.join(train_dir, filename),
            os.path.join(organized_dir, "dog", filename)
        )

print("تم التنظيم بنجاح")
print("عدد صور cat:", len(os.listdir(os.path.join(organized_dir, "cat"))))
print("عدد صور dog:", len(os.listdir(os.path.join(organized_dir, "dog"))))