# app.py
import uvicorn
from uvicorn import asgi

import streamlit as st
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import pickle
import numpy as np

def property_str_to_int(text):
        d = {'Single Family':1, 'Two Family':2, 'Three Family':3, 'Four Family':4,
           'Condo':8,'Residential':5}
        return d[text]

def str_to_int(text):
    d = {'Three rooms':3, 'Four rooms':4, 'Two rooms':2, 'Six rooms':6,
       'Eight rooms':8}
    return d[text]
    

# Load the trained pipeline
with open('trained_pipeline.pkl', 'rb') as file:
    loaded_pipeline = pickle.load(file)

# FastAPI app setup
app = FastAPI()

# CORS middleware for allowing requests from Streamlit app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Streamlit app
st.title("CSV Prediction App")

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
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
    # Make predictions using the loaded pipeline
    predictions = loaded_pipeline.predict(data)

    # Display predictions in Streamlit
    st.write("Predictions:")
    st.write(predictions)

    # Save predictions to a CSV file
    predictions_df = pd.DataFrame({'Predictions': predictions})
    predictions_csv = BytesIO()
    predictions_df.to_csv(predictions_csv, index=False)
    st.download_button("Download Predictions", predictions_csv.getvalue(), file_name="predictions.csv", key="predictions_csv")

# FastAPI endpoint for receiving CSV file and providing predictions
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    data = pd.read_csv(BytesIO(contents))

    # Make predictions using the loaded pipeline
    predictions = loaded_pipeline.predict(data)

    # Save predictions to a CSV file
    predictions_df = pd.DataFrame({'Predictions': predictions})
    predictions_csv = BytesIO()
    predictions_df.to_csv(predictions_csv, index=False)

    return FileResponse(predictions_csv, media_type="text/csv", filename="predictions.csv")
if __name__ == "__main__":
    # Run the FastAPI application using uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)