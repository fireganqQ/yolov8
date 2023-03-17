from ultralytics import YOLO
import cv2
import cvzone
import math, numpy as np

model = YOLO("yolov8n.pt") # dataset i tanımlıyoruz

classNames = model.names # dataset içindeki tanımlı obje ismi listesi

camera = cv2.VideoCapture(0)

while 1:
    _, cam = camera.read()

    result = model(cam, stream=True) # object detection işleminin başlatıldığı kod

    for r in result:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100 # confidence hesaplama
            print("-"*10,conf, "\n"*2)
            cls = int(box.cls[0])
            if conf>= 0.5: # Confidence değeri %50 den altını kabul etmiyoruz.
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(cam,(x1,y1),(x2,y2),(155,0,155),2)
                w, h = x2 - x1, y2 - y1
                cvzone.putTextRect(cam, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2)
                # cvzone.cornerRect(cam, (x1, y1, w, h))
            
            print("*"*10, cls)
    cv2.imshow("Frame", cam)
    key = cv2.waitKey(1)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()