from math import e
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Medical Insurance Charges Prediction", page_icon="💰", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Medical Insurance Charges Prediction</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: purple;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

model = pickle.load(open('model.pkl', 'rb'))


age = st.text_input("Enter your Age: ")
bmi = st.text_input("Enter your BMI: ")
children = st.slider("Children", 0, 5, 1)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Predict Charges"):

    input_df = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex": [sex],
        "smoker": [smoker],
        "region": [region]
    })


    prediction = model.predict(input_df)

    st.success(f"Estimated Insurance Charges: ₹ {prediction[0]:,.2f}")

else:
    st.warning("Please enter valid inputs to get a prediction.")
