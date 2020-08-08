import numpy as np
from PIL import Image, ImageOps
import logging
import operator

logging.basicConfig(level=logging.INFO)


def simple_inference(uploaded_image_path,
                     landmark_ids_names_mapping_dict,
                     loaded_models_list,
                     indices_to_class_labels_dict_list):
    '''
    Doing simple inference for single uploaded image.
    :param indices_to_class_labels_dict_list:
    :param loaded_models_list:
    :param landmark_ids_names_mapping_dict:
    :param uploaded_image_path: string, path to the uploaded image
    '''

    image = load_image(uploaded_image_path)
    predicted_id_confidence_score_dict = predict_class_confidence_score(image,
                                                                        loaded_models_list,
                                                                        indices_to_class_labels_dict_list)
    # sorting dictionary by values
    sorted_predicted_id_confidence_score_tuple_list = sorted(predicted_id_confidence_score_dict.items(),
                                                             key=operator.itemgetter(1), reverse=True)

    predicted_id_list = []
    predicted_name_list = []
    confidence_score_list = []
    for i in range(len(sorted_predicted_id_confidence_score_tuple_list)):
        predicted_id = sorted_predicted_id_confidence_score_tuple_list[i][0]
        confidence_score = sorted_predicted_id_confidence_score_tuple_list[i][1]
        predicted_name = landmark_ids_names_mapping_dict[int(predicted_id)]

        predicted_name_list.append(predicted_name)
        logging.info("Identified landmarkID : {}, Name {}".format(predicted_id, predicted_name))
        predicted_id_list.append(predicted_id)
        confidence_score_list.append(confidence_score)

    return confidence_score_list, predicted_id_list, predicted_name_list


def resize_image(image_path, new_width, new_height):
    image = Image.open(image_path, mode='r')
    image = ImageOps.fit(image, (new_width, new_height), Image.ANTIALIAS)
    return image


def load_image(uploaded_image_path):
    image = resize_image(uploaded_image_path, 224, 224)
    np_image = np.array(image)
    np_image = np_image / 255.0
    image = np.expand_dims(np_image, axis=0)
    return image


def predict_class_confidence_score(image, loaded_models_list, indices_to_class_labels_dict_list):
    results_list = []
    predicted_id_confidence_score_dict = {}
    for j in range(len(loaded_models_list)):
        results_list.append(loaded_models_list[j].predict(image))

        predicted_class = indices_to_class_labels_dict_list[j][np.argmax(results_list[j])]
        confidence_score = max(results_list[j][0]) * 100
        predicted_id_confidence_score_dict[predicted_class] = confidence_score

        logging.info("Identified landmark ID for image is : {}".format(predicted_class))
        logging.info("Confidence score : {}".format(confidence_score))
        logging.info("Whole Result array : {}\n".format(sorted(results_list[j][0])[-5:]))
    logging.info(
        "Predicted LandmarkId and Confidence score dictionary : \n {}".format(predicted_id_confidence_score_dict))
    return predicted_id_confidence_score_dict
