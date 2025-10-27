from ultralytics import YOLO

model = YOLO('yolo11m.pt')

if __name__ == '__main__':
    results = model.train(
        data='D:\Programming\Playground\datasets\Fly_Mos_Formatted\dataset.yaml',
        epochs=170,
        imgsz=640,
        batch=14
    )

    print("Training complete! Model saved.")