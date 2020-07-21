import numpy as np
import pandas as pd
from PIL import Image, ImageOps


def simple_inference(loaded_model, uploaded_image_path):

    '''
    Doing simple inference for single uploaded image.

    :param loaded_model: CNN model
    :param uploaded_image_path: string, path to the uploaded image
    '''

    with open("indices_to_class_labels_dict_group_5.json", "rb") as pickle_file:
        indices_to_class_labels_dict = pd.read_pickle(pickle_file, compression=None)

    image = resize_image(uploaded_image_path, 224, 224)

    np_image = np.array(image)
    np_image = np_image/255.0
    image = np.expand_dims(np_image, axis=0)
    result_array = loaded_model.predict(image)
    confidence_score = max(result_array[0]) * 100
    predicted_class = indices_to_class_labels_dict[np.argmax(result_array)]
    return confidence_score, predicted_class

def resize_image(image_path, new_width, new_height):
    image = Image.open(image_path)
    image = ImageOps.fit(image, (new_width, new_height), Image.ANTIALIAS)
    return image