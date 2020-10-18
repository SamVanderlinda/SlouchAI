"""
Pre-Req
Must export GOOGLE_APPLICATION_CREDENTIALS=key-file-path
or set GOOGLE_APPLICATION_CREDENTIALS=key-file-path on command-line
install google-cloud-automl, numpy, opencv-python, playsound
​
TODO()
Incorportate Background Filter
Add alert, visual and/or audio notification
"""

from playsound import playsound
from time import time
from google.cloud import automl  # pip install google-cloud-automl
import os  # for getting the path of the file
import cv2  # pip install numpy, then opencv-python, ?maybe bindings?
import tkinter  # for alerts
top = tkinter.Tk()  # for alerts
# Code to add widgets will go here...
# top.mainloop()

project_id = "imposing-gadget-292822"
model_id = "ICN6956427550008541184"

# Create a new VideoCapture object
cam = cv2.VideoCapture(0)
path = os.path.dirname(os.path.abspath(__file__)) + '\\' # windows
#path = os.path.dirname(os.path.abspath(__file__)) + '/' # macOS/linux

prediction_client = automl.PredictionServiceClient()

# Get the full path of the model.
model_full_id = automl.AutoMlClient.model_path(
    project_id, "us-central1", model_id
)

# Initialise variables to store current time difference as well as previous time call value
previous = time()
delta = 0

# Keep looping
while True:
    # Get the current time, increase delta and update the previous variable
    current = time()
    delta += current - previous
    previous = current


    # photo every 20 seconds
    if delta > 5:
        # Operations on image
        s, img = cam.read()

        file_name = f"filename{current}.jpg"
        cv2.imwrite(file_name,img) #save image
        print(path + file_name)
        with open(path + file_name, "rb") as content_file:
            content = content_file.read()

        os.remove(file_name)
        image = automl.Image(image_bytes=content)
        payload = automl.ExamplePayload(image=image)

		# probably needs to be differet
        params = {"score_threshold": "0.5"}

        request = automl.PredictRequest(
            name=model_full_id,
            payload=payload,
            params=params
        )
        response = prediction_client.predict(request=request)

        print("Prediction results:")
        for result in response.payload:
            print("Predicted class name: {}".format(result.display_name))
            print("Predicted class score: {}".format(result.classification.score))

		# if slouching is likely trigger alert
        if result.display_name == 'slouched':
            playsound('caralarm.mp3')

        delta = 0  # reset the timer