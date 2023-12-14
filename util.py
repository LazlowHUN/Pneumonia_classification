import base64
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import ImageOps, Image

def classify(image, model, class_names):
    img = tf.image.resize(image, [128, 128])

    img_array=tf.keras.utils.img_to_array(img)
    img_array=tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    class_name = class_names[0 if predictions[0][0] <= 0.35 else 1]
    accuracy_score = max(predictions[0][0], 1 - predictions[0][0]) * 100

    return class_name, accuracy_score


def save_to_csv(class_name, percentage, csv_file):
    new_data = [{'Class': class_name, 'Percentage': percentage}]

    new_df = pd.DataFrame(new_data)
    df = pd.read_csv(csv_file)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(csv_file, index=False)