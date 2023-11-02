import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify, save_to_csv
import pandas as pd

st.sidebar.title("Previous results:")

df = pd.read_csv("prev_results/prev.csv")
st.sidebar.write(df, header=False)

#set title

st.title("Pneumonia classification")


#set header

st.header("Please upload a chest X-ray image")


#upload file

file = st.file_uploader("", type=['jpeg', 'jpg', 'png'])


#load classifier

model = load_model('./model3.h5')


#class names

class_names = ['NORMAL', 'PNEUMONIA']


#display image

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    #classify image
    class_name, conf_score = classify(image, model, class_names)

    #write classification
    st.write("## {}". format(class_name))
    st.write("### score: {}". format(conf_score))

    save_to_csv(class_name, conf_score, "prev_results/prev.csv")