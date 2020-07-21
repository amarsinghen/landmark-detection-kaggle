import tensorflow as tf
import numpy as np

def get_delf(args):
	images, delf = args[0], args[1]
	results = []
	for i in range (0, len(images)):
		result.append(run_delf(images[i], delf))
	return results

def run_delf(image, delf):
	np_image = np.array(image)
	float_image = tf.image.convert_image_dtype(np_image, tf.float32)
	return delf(
		image=float_image, 
		score_threshold=tf.constant(100.0), 
		image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]), 
		max_feature_num=tf.constant(1000))