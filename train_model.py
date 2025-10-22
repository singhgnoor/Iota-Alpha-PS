from ultralytics import YOLO

model = YOLO('yolov8m.pt')

if __name__ == '__main__':
    # Train the model using your 'dataset.yaml' file
    results = model.train(
        data='D:\Programming\Playground\datasets\Fly_Mos_Formatted\dataset.yaml',  # Path to your yaml file
        epochs=50,                          # Start with 100 epochs
        imgsz=640,                           # Image size
        batch=16                              # Adjust based on your GPU (4 or 16)
    )

    print("Training complete! Model saved.")