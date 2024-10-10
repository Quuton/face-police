import gradio as gr
import cv2
from ultralytics import YOLO
model = YOLO(f'{config.OUTPUT_DIR}best.pt')

def get_prediction_frames(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_3channel = cv2.merge([gray_image, gray_image, gray_image])
    results = model(gray_3channel)
    return results[0].plot()

def main():
    demo = gr.Interface(
    get_prediction_frames,
    gr.Image(sources=["webcam"], streaming=True),
    "image",
    live=True
    )

    demo.launch()

if __name__ == "__main__":
    main()
