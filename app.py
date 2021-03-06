'''
Flask backend for predicting image landmark.
'''

import os

# Since there is no training, we want to run this code only on cpu save gpu resources
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import gzip
import pandas as pd

from inference import simple_inference
from flask import Flask, request, render_template, send_from_directory
from flask_swagger_ui import get_swaggerui_blueprint

# Clearing keras session in the beginning
tf.keras.backend.clear_session()

# Set root dir
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define Flask app
app = Flask(__name__)


# Define apps home page
@app.route("/")  # www.landmarks_recognition.com/
def index():
    return render_template("index.html")


# load landmarkIds to names mapping file
def landmark_ids_to_names_mapping():
    with gzip.open('data/landmark_ids_names_dict.pkl.gzip', 'rb') as pickle_file:
        landmark_ids_names_mapping_dict = pd.read_pickle(pickle_file)
    return landmark_ids_names_mapping_dict


# loading model weights and labels
def load_model_weights_and_class_labels():
    num_of_models = 2
    loaded_models_list = []
    indices_to_class_labels_dict_list = []

    for i in range(num_of_models):
        filepath = "model_weights/group{}_set224_resnet50_weights.hdf5".format(i + 10)
        loaded_models_list.append(tf.keras.models.load_model(filepath))
        with open("model_weights/group{}_indices_to_class_labels_dict.json".format(i + 10), "rb") as pickle_file:
            indices_to_class_labels_dict_list.append(pd.read_pickle(pickle_file))
    return loaded_models_list, indices_to_class_labels_dict_list


# Define upload function
@app.route("/upload", methods=["POST"])
def upload():
    upload_dir = os.path.join(APP_ROOT, "uploads")

    if not os.path.isdir(upload_dir):
        os.mkdir(upload_dir)

    img_name = ''
    for img in request.files.getlist("file"):
        img_name = img.filename
        destination = os.path.join(upload_dir, img_name)
        img.save(destination)

    landmark_ids_to_names = landmark_ids_to_names_mapping()
    loaded_models_list, indices_to_class_labels_dict_list = load_model_weights_and_class_labels()

    # inference
    confidence_score_list, predicted_id_list, predicted_name_list = simple_inference(os.path.join(upload_dir, img_name),
                                                                                     landmark_ids_to_names,
                                                                                     loaded_models_list,
                                                                                     indices_to_class_labels_dict_list)
    return render_template("result.html",
                           image_name=img_name,
                           confidence_score=confidence_score_list,
                           landmark_id_result=predicted_id_list,
                           landmark_name_result=predicted_name_list)


# Define helper function for finding image paths
@app.route("/upload/<filename>")
def send_image(filename):
    return send_from_directory("uploads", filename)


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        "app_name": "Landmark-Detection-Flask-Swagger"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Start the application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
