from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8l.pt')

# Use 0 for the default webcam
# You can also use a path to a video file instead, e.g., 'my_video.mp4'
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        # stream=True is more memory-efficient for video feeds
        results = model(frame, stream=True)

        # Visualize the results on the frame
        annotated_frame = next(results).plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()