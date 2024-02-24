# streamlit_app.py

import streamlit as st
import requests
import numpy as np
import pandas as pd


st.title("Streamlit CSV Prediction App")

def property_str_to_int(text):
        d = {'Single Family':1, 'Two Family':2, 'Three Family':3, 'Four Family':4,
           'Condo':8,'Residential':5}
        return d[text]

def str_to_int(text):
    d = {'Three rooms':3, 'Four rooms':4, 'Two rooms':2, 'Six rooms':6,
       'Eight rooms':8}
    return d[text]

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Display the uploaded data
    df = pd.read_csv(uploaded_file)
    
    
    
    cols = ['crime_rate', 'renovation_level', 'num_rooms', 'Property',
       'amenities_rating', 'carpet_area', 'property_tax_rate', 'Locality',
       'Residential', 'Estimated Value']
    df = df[cols]


    df['crime_rate'] = np.where(df['crime_rate']=='Not Provided',np.nan,df['crime_rate'])
    df['crime_rate'] = df['crime_rate'].astype(float)
    # app.py
    
    df['Property']= df['Property'].apply(property_str_to_int)
    df['num_rooms']= df['num_rooms'].apply(str_to_int)
    
    
    
    df['carpet_area'] = np.where(df['carpet_area'] == 'Not Provided', np.nan, df['carpet_area'])
    df['carpet_area'] = df['carpet_area'].astype(float)
    
    df['property_tax_rate'] = np.where(df['property_tax_rate'] == 'Not Provided', np.nan, df['property_tax_rate'])
    df['property_tax_rate'] = df['property_tax_rate'].astype(float)
    
    
    
    
    
    data = df
    st.write("Uploaded Data:")
    st.write(data)

    # Make a request to the FastAPI endpoint for predictions
    files = {'file': ('input.csv', uploaded_file, 'text/csv')}
    response = requests.post("http://127.0.0.1:8000/predict", files=files)

    # Display predictions
    if response.status_code == 200:
        predictions = pd.read_csv(pd.compat.StringIO(response.text))
        st.write("Predictions:")
        st.write(predictions)
    else:
        st.error(f"Error in making predictions. Status code: {response.status_code}")
