import functions_framework
from google.cloud import storage
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import requests
import json
import base64
import firebase_admin
from firebase_admin import credentials, db
from io import BytesIO
from PIL import Image
from datetime import datetime,timedelta

@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    request_json = request.get_json(silent=True)

    if request_json and 'animal' in request_json:
        base64_image = request_json['animal']
        image_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_data, np.uint8)

        utc_now = datetime.utcnow()
        colombo_time = utc_now + timedelta(hours=5, minutes=30)
        date = colombo_time.strftime("%Y-%m-%d")
        time = colombo_time.strftime("%H:%M:%S")

        storage_client = storage.Client()
        model_bucket_name = 'animal_classification_model'
        model_file_name = 'model.keras'
        bucket2 = storage_client.bucket(model_bucket_name)
        blob2 = bucket2.blob(model_file_name)
        local_path2 = f"/tmp/{model_file_name}"
        blob2.download_to_filename(local_path2)
        class_names = ['elephant', 'none', 'peacock', 'wildboar']
        IMAGE_SIZE = (128,128)
        model = tf.keras.models.load_model(local_path2)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        resized_image = tf.image.resize(image,IMAGE_SIZE)
        scaled_image = resized_image/255
        predictions = model.predict(np.expand_dims(scaled_image, axis=0))
        predicted_class_index = np.argmax(predictions)
        detected_animal = class_names[predicted_class_index]

        print(detected_animal)

        if not firebase_admin._apps:
            cred_bucket_name = 'firebaseserviceaccountjson'
            cred_file_name = 'claws-423416-firebase-adminsdk-qjk2q-ce055d7c5b.json'
            bucket3 = storage_client.bucket(cred_bucket_name)
            blob3 = bucket3.blob(cred_file_name)
            local_cred_path = '/tmp/serviceAccountKey.json'
            blob3.download_to_filename(local_cred_path)

            cred = credentials.Certificate(local_cred_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://claws-423416-default-rtdb.asia-southeast1.firebasedatabase.app/'
                })

        

        ref = db.reference('detected_animal')
        existing_data_ref = ref.child('animalInfo')
        existing_data_ref.update({
            'animal': detected_animal,
            'date': date,
            'image': base64_image,
            'time': time
        })

        ref = db.reference('response')
        existing_data_ref = ref.child('detectedAnimalName')
        existing_data_ref.update({
            'animalName': detected_animal,
            'detTime': time
        })

        
        return detected_animal
    else:
        return 'No image data found.', 400
        