from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")  # nano = fast on CPU

    def detect(self, frame):
        results = self.model(frame, conf=0.4, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                detections.append([x1, y1, x2, y2, cls, conf])

        return detections
