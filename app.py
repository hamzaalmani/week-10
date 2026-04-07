# app.py

import streamlit as st
import pandas as pd
from apputil import predict_rating

st.title("Coffee Rating Predictor")

price = st.number_input("Price (100g_USD)", min_value=0.0, value=10.0)

roast = st.text_input("Roast (e.g., Light, Medium, Dark)")

if st.button("Predict"):
    df = pd.DataFrame([[price, roast]], columns=["100g_USD", "roast"])
    pred = predict_rating(df)
    st.success(f"Predicted rating: {pred[0]:.2f}")