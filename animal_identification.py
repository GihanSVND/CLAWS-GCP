import functions_framework
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from google.cloud import storage


# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def hello_gcs(cloud_event):
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
    bucket1 = storage_client.bucket(bucket_name)
    blob1 = bucket1.blob(file_name)
    local_path1 = f"/tmp/{file_name}"
    blob.download_to_filename(local_path1)

    model_bucket_name = 'trained_classification_model'
    model_file_name = 'model.keras'
    bucket2 = storage_client.bucket(model_bucket_name)
    blob2 = bucket2.blob(model_file_name)
    local_path2 = f"/tmp/{model_file_name}"
    blob2.download_to_filename(local_path2)

    class_names = ['elephant', 'none', 'peacock', 'wildboar']
    model = keras.models.load_model(local_path)
    image = cv2.imread(local_path1)
    IMAGE_SIZE = (128,128)
    resized_image = tf.image.resize(image,IMAGE_SIZE)
    scaled_image = resized_image/255
    predictions = model.predict(np.expand_dims(scaled_image, axis=0))
    predicted_class_index = np.argmax(predictions)
    detected_animal = class_names[predicted_class_index]

    print(detected_animal)



