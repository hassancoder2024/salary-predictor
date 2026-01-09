import streamlit as st 
import numpy as np
import pickle 


model, scaler = pickle.load(open("linear_regression.pkl", "rb"))

st.title("Salary Predictor")

experience = st.number_input("Enter the years of experience")

hoursofwork = st.number_input("Enter the hours of work")

if st.button("Predict"):
    features = np.array([[experience, hoursofwork]])
    features_scaled = scaler.transform(features)
    results = model.predict(features_scaled)
    st.write("Predicted salary:", results)