from ultralytics.solutions import Heatmap
import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture(
        r"D:\Programming\Playground\Iota-Alpha-PS\Here’s How That Annoying Fly Dodges Your Swatter ｜ Deep Look [jBPFCvEhv9Y].mp4")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    output_path = "annotated_video_with_heatmap.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    heatmap_obj = Heatmap(
        model=r'D:\Programming\Playground\Iota-Alpha-PS\runs\detect\train11\weights\best.pt', # refined YOLO11m
        tracker="bytetrack.yaml",
        colormap=cv2.COLORMAP_JET,
    )

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = heatmap_obj(frame)

            annotated_frame = results.plot_im

            cv2.imshow("Plotting heatmap", cv2.resize(annotated_frame, (640, 480)))
            out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()