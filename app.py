import gradio as gr
import cv2
import requests
import os
import numpy as np
from ultralytics import YOLO
import math

model = YOLO('yolo-Weights/best.pt')

# object classes
classNames = ["book","lego","plush"]

def show_preds_image(image):
    if isinstance(image, str):  # If the input is a string, treat it as a file path
        image = cv2.imread(image)
    elif isinstance(image, np.ndarray):  # If the input is a numpy array, treat it as an image
        pass
    else:
        raise ValueError("Invalid input type for image")

    frame_copy = image.copy()
    #results=model.predict(source=image)
    outputs = model.predict(source=image)
    results = outputs[0].cpu().numpy()
    #print(results.boxes.xyxy)
    #print("end of results")
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            
            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)
            
            if confidence > 0.7:
                # put box in cam
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)


                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(image, classNames[cls], org, font, fontScale, color, thickness)

    yield image

inputs_video = [
    gr.Image(label="Input Video", sources="webcam"),
 
]
outputs_video = [
    gr.components.Image(type="numpy", label="Output Image"),
]
interface_video = gr.Interface(
    fn=show_preds_image,
    inputs=inputs_video,
    outputs=outputs_video,
    title="Plush-lego-book detector",
    #examples=video_path,
    cache_examples=False,
)

interface_video.queue().launch()