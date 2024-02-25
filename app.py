# streamlit_app.py
import streamlit as st
import requests
import pandas as pd
st.title("Streamlit-FastAPI Integration")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    st.write("File Uploaded!")

    response = requests.post("http://127.0.0.1:8000/predict", files={"file": uploaded_file})

    if response.status_code == 200:
        # Save the response content to a local file
        with open("predictions.csv", "wb") as f:
            f.write(response.content)
    
        # Display the local CSV file in Streamlit
        st.write("Predictions:")
        st.dataframe(pd.read_csv("predictions.csv"))
    else:
        st.error(f"Error processing the file. Status Code: {response.status_code}")
