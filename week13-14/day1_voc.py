import os
import xml.etree.ElementTree as ET

# ترتيب الفئات يحدد الرقم (index) اللي YOLO يستخدمه لكل فئة
classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

annotations_dir = r"C:\Users\A Center\cv_engineer_path\week14\NEU-DET\train\annotations"
labels_output_dir = r"C:\Users\A Center\cv_engineer_path\week14\NEU-DET\train\labels"

os.makedirs(labels_output_dir, exist_ok=True)

def convert_voc_to_yolo(xml_path, output_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)

    yolo_lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_id = classes.index(class_name)

        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # التحويل: من زوايا مطلقة إلى مركز+أبعاد نسبية
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    filename = os.path.splitext(os.path.basename(xml_path))[0]
    output_path = os.path.join(output_dir, filename + ".txt")
    with open(output_path, "w") as f:
        f.write("\n".join(yolo_lines))


# تحويل كل ملفات XML بالمجلد
xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
for xml_file in xml_files:
    convert_voc_to_yolo(os.path.join(annotations_dir, xml_file), labels_output_dir)

print(f"تم تحويل {len(xml_files)} ملف بنجاح")
print("أول ملف بعد التحويل:")
with open(os.path.join(labels_output_dir, "crazing_1.txt")) as f:
    print(f.read())