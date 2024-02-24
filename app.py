# streamlit_app.py
import streamlit as st
import requests

st.title("Streamlit-FastAPI Integration")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    st.write("File Uploaded!")

    # Assuming you want to send the file to FastAPI for processing
    response = requests.post("http://127.0.0.1:8000/predict", files={"file": uploaded_file})

    if response.status_code == 200:
        predictions = response.json()["predictions"]
        st.write("Predictions:")
        st.write(predictions)
    else:
        st.error(f"Error processing the file. Status Code: {response.status_code}")
