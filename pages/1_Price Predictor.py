import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analysis Model")

# Correct paths to your models
model_df_path = "/mount/src/real-state-model/models/df.pkl"
model_pipeline_path = "/mount/src/real-state-model/models/pipeline.pkl"

try:
    with open(model_df_path, 'rb') as file:
        df = pickle.load(file)
    
    with open(model_pipeline_path, 'rb') as file:
        pipeline = pickle.load(file)
except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}")
    st.stop()
except pickle.UnpicklingError:
    st.error("Error unpickling the model. Check version compatibility.")
    st.stop()

st.header('Enter your inputs')

# Ensure that the options match the training data
property_type = st.selectbox('Property Type', sorted(df['property_type'].unique().tolist()))
sector = st.selectbox('Sector', sorted(df['sector'].unique().tolist()))
bedrooms = float(st.selectbox('Number of Bedroom', sorted(df['bedRoom'].unique().tolist())))
bathroom = float(st.selectbox('Number of Bathrooms', sorted(df['bathroom'].unique().tolist())))
balcony = st.selectbox('Balconies', sorted(df['balcony'].unique().tolist()))
property_age = st.selectbox('Property Age', sorted(df['agePossession'].unique().tolist()))
built_up_area = float(st.number_input('Built Up Area'))
servant_room = float(st.selectbox('Servant Room', [0.0, 1.0]))
store_room = float(st.selectbox('Store Room', [0.0, 1.0]))
furnishing_type = st.selectbox('Furnishing Type', sorted(df['furnishing_type'].unique().tolist()))
luxury_category = st.selectbox('Luxury Category', sorted(df['luxury_category'].unique().tolist()))
floor_category = st.selectbox('Floor Category', sorted(df['floor_category'].unique().tolist()))

if st.button('Predict'):
    # Form a dataframe with correct column names
    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony', 'agePossession', 'built_up_area', 'servant_room', 'store_room', 'furnishing_type', 'luxury_category', 'floor_category']
    
    one_df = pd.DataFrame(data, columns=columns)
    
    # Predict
    try:
        # Ensure that the columns in one_df match the expected columns in the pipeline
        one_df = one_df[columns]
        base_price = np.expm1(pipeline.predict(one_df))[0]
        low = base_price - 0.22
        high = base_price + 0.22
        st.text(f"The price of the flat is between {round(low, 2)} Cr and {round(high, 2)} Cr")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
