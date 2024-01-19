import cv2
import inference
import supervision as sv

annotator = sv.BoxAnnotator()

def render(predictions, image):
    classes = {item["class_id"]: item["class"] for item in predictions["predictions"]}

    detections = sv.Detections.from_roboflow(predictions)

    print(predictions)

    image = annotator.annotate(
        scene=image, detections=detections, labels=[classes[i] for i in detections.class_id]
    )

    cv2.imshow("Prediction", image)
    cv2.waitKey(1)


inference.Stream(
    source="webcam",
    model="house-items/1",
    output_channel_order="BGR",
    use_main_thread=True,
    on_prediction=render,
    api_key="1lRYAtOHBXghetiCeXTZ"
)