import cv2
import numpy as np
import onnxruntime as ort

CONF_THRESHOLD = 0.5

# Load image
img = cv2.imread("static/image.jpg")
orig_h, orig_w = img.shape[:2]

# Preprocess
img_resized = cv2.resize(img, (640, 640))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_norm = img_rgb.astype(np.float32) / 255.0
img_input = np.transpose(img_norm, (2, 0, 1))
img_input = np.expand_dims(img_input, axis=0)

# Load ONNX model
session = ort.InferenceSession("model/yolo11n.onnx")
input_name = session.get_inputs()[0].name

# Inference
outputs = session.run(None, {input_name: img_input})
pred = outputs[0][0]  # shape: [84, 8400]

boxes = []

for i in range(pred.shape[1]):
    class_scores = pred[4:, i]
    cls_id = np.argmax(class_scores)
    conf = class_scores[cls_id]

    if conf < CONF_THRESHOLD:
        continue

    x, y, w, h = pred[0:4, i]

    x1 = int((x - w / 2) * orig_w / 640)
    y1 = int((y - h / 2) * orig_h / 640)
    x2 = int((x + w / 2) * orig_w / 640)
    y2 = int((y + h / 2) * orig_h / 640)

    boxes.append((cls_id, conf, [x1, y1, x2, y2]))

# Draw results
output_img = img.copy()
for cls_id, conf, (x1, y1, x2, y2) in boxes:
    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(output_img, f"{cls_id} {conf:.2f}",
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1)

cv2.imwrite("static/output_onnx.jpg", output_img)

# Print results
for cls_id, conf, bbox in boxes:
    print(f"Class: {cls_id}, Confidence: {conf:.2f}, BBox: {bbox}")
