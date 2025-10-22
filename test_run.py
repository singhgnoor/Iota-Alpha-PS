from ultralytics import YOLO
import cv2
from time import time

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    model = YOLO("yolo11m.pt")

    while cam.isOpened():
        start = time()
        ret, frame = cam.read()
        if ret:
            result = model(frame, stream=True, conf=.5)
            frame = next(result).plot()
            cv2.putText(frame, f"FPS : {round(1/(time()-start))}", (10, 30),  # Position (x, y)
                        cv2.FONT_HERSHEY_SIMPLEX, 1,  # Font and scale
                        (0, 255, 0), 2)  # Color (BGR) and thickness
            cv2.imshow('Yolo11', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Ended")
            break

        start = time()



