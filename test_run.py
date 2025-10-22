from ultralytics import YOLO
import cv2
from collections import defaultdict, deque
from time import sleep

FRAME_LIMIT = 150

if __name__ == '__main__':
    # Load the YOLOv8 model
    model = YOLO(r'D:\Programming\Playground\Iota-Alpha-PS\runs\detect\train6\weights\best.pt')

    cap = cv2.VideoCapture(
        r"D:\Programming\Playground\Iota-Alpha-PS\Here’s How That Annoying Fly Dodges Your Swatter ｜ Deep Look [jBPFCvEhv9Y].mp4")

    # --- START: Added Code for Video Saving ---
    # Get video properties (width, height, FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    output_path = "Annotated train 6 : Here’s How That Annoying Fly Dodges Your Swatter ｜ Deep Look.mp4"  # You can change this
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    # --- END: Added Code for Video Saving ---

    # To draw the trail of path
    track_history = defaultdict(lambda: deque(maxlen=FRAME_LIMIT))

    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame.
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")

            # Create a variable to hold the frame that will be saved
            # Default to the original frame
            frame_to_save = frame

            # We must loop over the generator to get the results for this frame.
            for result in results:

                # Now, 'result' is the object we were trying to access
                if result.boxes.id is not None:
                    # Get the boxes and track IDs
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()

                    # Visualize the results on the frame
                    annotated_frame = result.plot()

                    # Plot the tracks and update history
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track_history[track_id].append((int(x), int(y)))
                        points = track_history[track_id]
                        for i in range(1, len(points)):
                            if points[i - 1] is None or points[i] is None:
                                continue
                            cv2.line(annotated_frame, points[i - 1], points[i], (0, 255, 0), 2)

                    # Display the annotated frame (resized)
                    cv2.imshow("YOLOv8 Tracking", cv2.resize(annotated_frame, (640, 480)))

                    # Update the frame_to_save to the annotated one
                    frame_to_save = annotated_frame

                else:
                    # If no objects are detected/tracked, show original frame
                    cv2.imshow("YOLOv8 Tracking", cv2.resize(frame, (640, 480)))
                    # frame_to_save remains the original 'frame'

            # --- START: Modified Code for Video Saving ---
            # Write the frame (either original or annotated) to the output video
            # This writes the frame at its original resolution
            out.write(frame_to_save)
            # --- END: Modified Code for Video Saving ---

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # --- START: Modified Code for Releasing ---
    # Release the video capture and writer objects
    cap.release()
    out.release()  # Release the VideoWriter

    # Close all display windows
    cv2.destroyAllWindows()
    # --- END: Modified Code for Releasing ---