import os
import shutil

classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

def reorganize_labels(labels_flat_dir, labels_output_base):
    for class_name in classes:
        class_dir = os.path.join(labels_output_base, class_name)
        os.makedirs(class_dir, exist_ok=True)

    for filename in os.listdir(labels_flat_dir):
        if not filename.endswith('.txt'):
            continue
        # نحدد الفئة من اسم الملف نفسه (مثلاً crazing_1.txt → crazing)
        for class_name in classes:
            if filename.startswith(class_name):
                src = os.path.join(labels_flat_dir, filename)
                dst = os.path.join(labels_output_base, class_name, filename)
                shutil.move(src, dst)
                break

# إعادة تنظيم train/labels
reorganize_labels(
    r"C:\Users\A Center\cv_engineer_path\week14\NEU-DET\train\labels",
    r"C:\Users\A Center\cv_engineer_path\week14\NEU-DET\train\labels"
)

# إعادة تنظيم validation/labels
reorganize_labels(
    r"C:\Users\A Center\cv_engineer_path\week14\NEU-DET\validation\labels",
    r"C:\Users\A Center\cv_engineer_path\week14\NEU-DET\validation\labels"
)

print("تم تنظيم مجلدات labels لتطابق بنية images")