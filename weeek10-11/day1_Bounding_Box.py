def calculate_iou(box1, box2):
    # كل صندوق: (x_min, y_min, x_max, y_max)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union

# مثال: صندوقين متداخلين جزئيًا
box_true = (50, 50, 150, 150)   # الصندوق الصحيح (الحقيقي)
box_pred = (70, 70, 170, 170)   # الصندوق المتوقَّع من النموذج

iou = calculate_iou(box_true, box_pred)
print("IoU:", iou)