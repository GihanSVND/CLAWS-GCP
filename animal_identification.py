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
def main(cloud_event):
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

def detection(mod_add,img_add):
    class_names = ['elephant', 'none', 'peacock', 'wildboar']

    model_local_path = '/tmp/model.keras'
    download_blob(mod_add, model_local_path)
    model = keras.models.load_model(model_local_path)

    image_local_path = '/tmp/image.jpg'
    download_blob(img_add, image_local_path)
    
    image = cv2.imread(image_local_path)
    if image is None:
        raise ValueError("Failed to read the image from the local file path.")

    IMAGE_SIZE = (128,128)
    resized_image = tf.image.resize(image,IMAGE_SIZE)
    scaled_image = resized_image/255
    predictions = model.predict(np.expand_dims(scaled_image, axis=0))
    predicted_class_index = np.argmax(predictions)
    detected_animal = class_names[predicted_class_index]
    return detected_animal

def download_blob(gcs_url, local_path):
    # Parse the GCS URL
    if gcs_url.startswith("gs://"):
        gcs_url = gcs_url[5:]
    bucket_name, blob_name = gcs_url.split("/", 1)
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)