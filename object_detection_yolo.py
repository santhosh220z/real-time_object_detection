import torch
import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

def draw_boxes(image, boxes):
    print(boxes) 

    for box in boxes:
        print(box)  
        x1, y1, x2, y2 = box[:4]  
        conf = box[4] if len(box) > 4 else 0  
        cls = box[5] if len(box) > 5 else 0 

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 

        label = f'{model.names[int(cls)]} {conf:.2f}' 
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img)
    if isinstance(results, list):
        boxes = results[0].boxes.xyxy.numpy()  
    else:
        boxes = results.boxes.xyxy.numpy()  

    frame_with_boxes = draw_boxes(frame, boxes)

    cv2.imshow("YOLOv8 Object Detection", frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
