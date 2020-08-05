'''
Flask backend for predicting image landmark.
'''

# Import dependencies
import os
# Since there is no training, we want to run this code only on cpu save gpu resources
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
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


# Define upload function
@app.route("/upload", methods=["POST"])
def upload():
    upload_dir = os.path.join(APP_ROOT, "uploads/")

    if not os.path.isdir(upload_dir):
        os.mkdir(upload_dir)

    img_name = ''
    for img in request.files.getlist("file"):
        img_name = img.filename
        destination = os.path.join(upload_dir, img_name)
        img.save(destination)

    # inference
    confidence_score, predicted_class = simple_inference(os.path.join(upload_dir, img_name))
    return render_template("result.html", image_name=img_name, confidence_score=confidence_score,
                           landmark_id_result=predicted_class)


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
