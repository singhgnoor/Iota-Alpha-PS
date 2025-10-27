""" Just testing real-time object detection from pre-trained YOLO"""
from ultralytics import YOLO
import cv2

model = YOLO('yolov8l.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame, stream=True)
        annotated_frame = next(results).plot()
        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()