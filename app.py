'''
Flask backend for image-to-image search pipeline.
'''

#Import dependencies
import os
# Since there is no training, we want to run this code only on cpu save gpu resources
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from inference import simple_inference

#import Flask dependencies
from flask import Flask, request, render_template, send_from_directory

# Clearing keras session in the beginning
tf.keras.backend.clear_session()

#Set root dir
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

#Load model
trained_model = tf.keras.models.load_model('group5_set224_resnet50_NO_imagenet_weights_07112020.h5')

#Define Flask app
app = Flask(__name__, static_url_path='/static')

#Define apps home page
@app.route("/") #www.landmarks_recognition.com/
def index():
	return render_template("index.html")

#Define upload function
@app.route("/upload", methods=["POST"])
def upload():

	upload_dir = os.path.join(APP_ROOT, "uploads/")

	if not os.path.isdir(upload_dir):
		os.mkdir(upload_dir)

	for img in request.files.getlist("file"):
		img_name = img.filename
		destination = os.path.join(upload_dir, img_name)
		img.save(destination)

	#inference
	confidence_score, predicted_class = simple_inference(trained_model, os.path.join(upload_dir, img_name))
	return render_template("result.html", image_name=img_name, confidence_score=confidence_score, landmark_id_result=predicted_class)

#Define helper function for finding image paths
@app.route("/upload/<filename>")
def send_image(filename):
	return send_from_directory("uploads", filename)

#Start the application

if __name__ == "__main__":
	app.run(port=5000, debug=True)