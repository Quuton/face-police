from ultralytics import YOLO
import config
import os

def main():
    model = YOLO('yolov8n.pt')
    model.train(data=os.path.join(config.PROCESS_DIR, 'data.yaml'), epochs=config.TRAIN_EPOCHS, imgsz=416, batch=config.TRAIN_BATCH_SIZE, cache=True)

if __name__ == "__main__":
    main()