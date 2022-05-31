# import necessary libraries
import flask
import statistics
import io
import string
import time
import os
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request

# load CT scan models
def ct_scan_models():
    # load inception models
    inception = tf.keras.models.load_model('/content/drive/MyDrive/model_ct_scan/final_batch_64_preprocess/InceptionV3.h5')
    # load DenseNet201 models
    densenet = tf.keras.models.load_model('/content/drive/MyDrive/model_ct_scan/final_batch_64_preprocess/DenseNet201.h5')
    # Efficient Net v25
    efficient_net = tf.keras.models.load_model('/content/drive/MyDrive/model_ct_scan/final_batch_64_preprocess/EfficientNetV2S.h5')

    model_dict = {}
    model_dict['inception'] = inception
    model_dict['densenet'] = densenet
    model_dict['efficient_net'] = efficient_net

    return model_dict

    
# load X-ray models
def x_ray_models():
    # load inception models
    inception = tf.keras.models.load_model('/content/drive/MyDrive/model_chest_xray/InceptionResNetV2.h5')
    # load DenseNet201 models
    densenet = tf.keras.models.load_model('/content/drive/MyDrive/model_chest_xray/DenseNet201.h5')
    # Efficient Net v25
    efficient_net = tf.keras.models.load_model('/content/drive/MyDrive/model_chest_xray/EfficientNetV2S.h5')

    model_dict = {}
    model_dict['inception'] = inception
    model_dict['densenet'] = densenet
    model_dict['efficient_net'] = efficient_net

    return model_dict

#Preload models
models_ct=ct_scan_models()
models_xray=x_ray_models()


# Read image
def read_image_from_request():
    if 'file' not in request.files:
        return "File request not found"
    file = request.files.get('file')

    file.save(os.path.join('images', 'upload_image.jpg'))


def prepare_image():
    # Read image from URL
    read_image_from_request()
    img = cv2.imread(os.path.join('images', 'upload_image.jpg'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, 0)
    return img

def predict_results(models, img):
    inception_res = models['inception'].predict(img)
    inception_res = np.argmax(inception_res, axis=-1)

    dense_res = models['densenet'].predict(img)
    dense_res = np.argmax(dense_res, axis=-1)

    efficient_res = models['efficient_net'].predict(img)
    efficient_res = np.argmax(efficient_res, axis=-1)

    max_voting_res = np.array([])
    max_voting_res = np.append(
                            max_voting_res, 
                            int(statistics.mode([ int(inception_res),  int(dense_res) , int(efficient_res) ])) 
                            )

    prediction=np.int64(max_voting_res)[0]

    print(prediction)
    return "Covid" if prediction==0 else "Non-Covid"



# Register the Flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def getter():
    return jsonify(prediction="COVID-19 PREDICTION")

# Register ct route
@app.route('/ct', methods=['POST'])
def ct_scan_prediction():
    file = request.files.get('file')
    if not file:
        return jsonify(Status=400, message='Image not found')

    # Prepare the image
    img = prepare_image()

    
    # load CT scan models
    models = models_ct

    # prediction
    prediction = predict_results(models,img)

    return jsonify(prediction=prediction, Status=200, Type = "CT-Scan")

# Register xray route
@app.route('/xray', methods=['POST'])
def xray_prediction():
    file = request.files.get('file')
    if not file:
        return jsonify(Status=400, message='Image not found')

    # Prepare the image
    img = prepare_image()

    # load CT scan models
    models = models_xray

    # prediction
    prediction = predict_results(models,img)

    return jsonify(prediction=prediction, status=200, Type='X-ray')
