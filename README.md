Object Detection App

This is a simple object detection application built with Python, Gradio, and YOLOv. It allows you to upload an image and it will return the same image with bounding boxes and labels for detected objects.

Installation

Before running the app, you need to install the required libraries. You can do this by running the following command:

pip install -r requirements.txt

This will install all the libraries listed in the requirements.txt file.

Usage

To start the app, run the following command:

python app.py

This will start the Gradio interface. You can access the interface by opening a web browser and navigating to the URL displayed in the terminal.

Once the interface is open, you can upload an image by clicking the "Browse" button and selecting an image file from your computer. After you've selected an image, click the "Submit" button to process the image.

The app will display the processed image with bounding boxes and labels for detected objects. The bounding boxes are drawn in red, and the labels are drawn in green above the corresponding bounding box.

Expected Results

The app uses the YOLOv5 model for object detection, which can detect various types of objects. The exact types of objects that can be detected depend on the classes that the model was trained on.

The app will only draw bounding boxes and labels for objects with a confidence of 0.7 or higher. This threshold can be adjusted in the app.py file.