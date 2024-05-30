import functions_framework
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def hello_gcs(cloud_event):
    data = cloud_event.data

    event_id = cloud_event["id"]
    event_type = cloud_event["type"]

    bucket = data["bucket"]
    name = data["name"]
    metageneration = data["metageneration"]
    timeCreated = data["timeCreated"]
    updated = data["updated"]

    image_path = f"gs://{bucket}/{name}"  # Construct the full path
    model_path = f"gs://trained_classification_model/model.keras"
    animal = detection(model_path,image_path)
   
    print(f"Event ID: {event_id}")
    print(f"Event type: {event_type}")
    print(f"Bucket: {bucket}")
    print(f"File: {name}")
    print(f"Metageneration: {metageneration}")
    print(f"Created: {timeCreated}")
    print(f"Updated: {updated}")
    print(animal)


def detection(mod_add,img_add):
    class_names = ['elephant', 'none', 'peacock', 'wildboar']
    model = keras.models.load_model(mod_add)
    image = cv2.imread(img_add)
    IMAGE_SIZE = (128,128)
    resized_image = tf.image.resize(image,IMAGE_SIZE)
    scaled_image = resized_image/255
    predictions = model.predict(np.expand_dims(scaled_image, axis=0))
    predicted_class_index = np.argmax(predictions)
    detected_animal = class_names[predicted_class_index]
    return detected_animal