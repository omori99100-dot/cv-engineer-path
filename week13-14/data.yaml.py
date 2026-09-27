yaml_content = """train: C:/Users/A Center/cv_engineer_path/week14/NEU-DET/train/images
val: C:/Users/A Center/cv_engineer_path/week14/NEU-DET/validation/images

nc: 6
names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
"""

yaml_path = r"C:\Users\A Center\cv_engineer_path\week14\NEU-DET\data.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("تم إنشاء data.yaml بنجاح")
