from ultralytics import YOLO
import cv2
from collections import defaultdict, deque

FRAME_LIMIT = 150

if __name__ == '__main__':
    model = YOLO(r'D:\Programming\Playground\Iota-Alpha-PS\runs\detect\train11\weights\best.pt') # Refined yolo11m

    cap = cv2.VideoCapture(
        r"D:\Programming\Playground\Iota-Alpha-PS\Here’s How That Annoying Fly Dodges Your Swatter ｜ Deep Look [jBPFCvEhv9Y].mp4")

    # Get video properties (width, height, FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    output_path = "annotated_video_with_heatmap.mp4"  # --- Changed output name ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # To draw the trail of path
    track_history = defaultdict(lambda: deque(maxlen=FRAME_LIMIT))

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")

            annotated_frame = frame.copy()
            for result in results:
                if result.boxes.id is not None:
                    # Get the boxes and track IDs
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()

                    # Visualize the results on the frame
                    annotated_frame = result.plot()

                    # Plot the tracks and update history
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box

                        # Append the new center point
                        track_history[track_id].append((int(x), int(y)))

                        # Draw the tracking path
                        points = track_history[track_id]
                        for i in range(1, len(points)):
                            if points[i - 1] is None or points[i] is None:
                                continue
                            cv2.line(annotated_frame, points[i-1], points[i], (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("YOLOv8 Tracking", cv2.resize(annotated_frame, (640, 480)))
            out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # Release the video capture object and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()