# app/main.py

import cv2
import numpy as np
import tensorflow as tf

# Load saved model (update path if needed)
model = tf.saved_model.load("model/ssd_mobilenet_v2_fpnlite_320x320/saved_model")
infer = model.signatures['serving_default']

# Load label map for COCO dataset
LABELS = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
          6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light'}

# Set up webcam
cap = cv2.VideoCapture(0)  # Change to a video file path if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = infer(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    class_ids = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    height, width, _ = frame.shape

    for i in range(len(scores)):
        if scores[i] > 0.5:
            y1, x1, y2, x2 = boxes[i]
            (left, top, right, bottom) = (x1 * width, y1 * height, x2 * width, y2 * height)
            label = LABELS.get(class_ids[i], "Unknown")
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {int(scores[i]*100)}%',
                        (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    cv2.imshow('Real-Time Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
