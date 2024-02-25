# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
import numpy as np
import io
import traceback
import json
from fastapi.responses import StreamingResponse
import csv
from fastapi.responses import StreamingResponse
import tempfile
import shutil

app = FastAPI()

# Load the trained model
with open('prediction_pipeline.pkl', 'rb') as file:
    loaded_pipeline = pickle.load(file)

def property_str_to_int(text):
    d = {'Single Family': 1, 'Two Family': 2, 'Three Family': 3, 'Four Family': 4, 'Condo': 8, 'Residential': 5}
    return d.get(text, 0)

def str_to_int(text):
    d = {'Three rooms': 3, 'Four rooms': 4, 'Two rooms': 2, 'Six rooms': 6, 'Eight rooms': 8}
    return d.get(text, 0)




@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        cols = ['crime_rate', 'renovation_level', 'num_rooms', 'Property',
                'amenities_rating', 'carpet_area', 'property_tax_rate', 'Locality',
                'Residential', 'Estimated Value']
        df = df[cols]

        df['crime_rate'] = np.where(df['crime_rate'] == 'Not Provided', np.nan, df['crime_rate'])
        df['crime_rate'] = df['crime_rate'].astype(float)

        df['Property'] = df['Property'].apply(property_str_to_int)
        df['num_rooms'] = df['num_rooms'].apply(str_to_int)

        df['carpet_area'] = np.where(df['carpet_area'] == 'Not Provided', np.nan, df['carpet_area'])
        df['carpet_area'] = df['carpet_area'].astype(float)

        df['property_tax_rate'] = np.where(df['property_tax_rate'] == 'Not Provided', np.nan, df['property_tax_rate'])
        df['property_tax_rate'] = df['property_tax_rate'].astype(float)

        predictions = loaded_pipeline.predict(df)

    
        df = pd.DataFrame({'Predictions':predictions})

        # Write the DataFrame to a CSV file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            df.to_csv(temp_file.name, index=False)

        # Return the CSV file as a streaming response
        return StreamingResponse(open(temp_file.name, "rb"), media_type="text/csv", headers={"Content-Disposition": "attachment;filename=predictions.csv"})

        # main.py

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"An error occurred: {str(e)}\nTraceback: {traceback_str}")
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)


  



        