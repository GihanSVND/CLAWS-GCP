import functions_framework
from google.cloud import storage
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import requests
import json

# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def main(cloud_event):
    data = cloud_event.data

    event_id = cloud_event["id"]
    event_type = cloud_event["type"]

    bucket_name = data["bucket"]
    file_name = data["name"]
    metageneration = data["metageneration"]
    time_created = data["timeCreated"]
    updated = data["updated"]
    
    print(f"Event ID: {event_id}")
    print(f"Event type: {event_type}")
    print(f"Bucket: {bucket_name}")
    print(f"File: {file_name}")
    print(f"Metageneration: {metageneration}")
    print(f"Created: {time_created}")
    print(f"Updated: {updated}")

    storage_client = storage.Client()
    
    # Download the image file
    bucket1 = storage_client.bucket(bucket_name)
    blob1 = bucket1.blob(file_name)
    local_path1 = f"/tmp/{file_name}"
    blob1.download_to_filename(local_path1)

    # Load the classification model
    model_bucket_name = 'trained_classification_model'
    model_file_name = 'model.keras'
    bucket2 = storage_client.bucket(model_bucket_name)
    blob2 = bucket2.blob(model_file_name)
    local_path2 = f"/tmp/{model_file_name}"
    blob2.download_to_filename(local_path2)

    class_names = ['elephant', 'none', 'peacock', 'wildboar']
    IMAGE_SIZE = (128,128)

    model = keras.models.load_model(local_path2)
    image = cv2.imread(local_path1)

    resized_image = tf.image.resize(image,IMAGE_SIZE)
    scaled_image = resized_image/255
    
    predictions = model.predict(np.expand_dims(scaled_image, axis=0))

    predicted_class_index = np.argmax(predictions)
    detected_animal = class_names[predicted_class_index]

    print(detected_animal)

    if detected_animal != "none":
        destination_bucket_name = 'detected-animals'
        destination_bucket = storage_client.bucket(destination_bucket_name)
        new_blob = bucket1.copy_blob(blob1, destination_bucket)

    # Send the detected animal data to the specified endpoint
    url = 'http://127.0.0.1:5000/endpoint'
    data = {"animal": detected_animal}

    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)

    print(response.status_code, response.reason)
