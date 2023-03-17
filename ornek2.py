from ultralytics import YOLO

model = YOLO("yolov8n.pt") # dataset i tanımlıyoruz

model.predict(source=0, show=True) # object detection işleminin başlatıldığı kod