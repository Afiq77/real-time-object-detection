# utils/detection_utils.py

import cv2

# Basic COCO label mapping (expand as needed)
LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light'
}

def draw_boxes(frame, boxes, class_ids, scores, threshold=0.5):
    height, width, _ = frame.shape

    for i in range(len(scores)):
        if scores[i] > threshold:
            y1, x1, y2, x2 = boxes[i]
            (left, top, right, bottom) = (x1 * width, y1 * height, x2 * width, y2 * height)
            label = LABELS.get(class_ids[i], "Unknown")

            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {int(scores[i]*100)}%',
                        (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
    return frame
