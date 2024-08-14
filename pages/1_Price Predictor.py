import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn

st.set_page_config(page_title="Analysis Model")

file_paths = {
    "df": "/mount/src/real-state-model/models/df.pkl",
    "pipeline": "/mount/src/real-state-model/models/pipeline.pkl"
}

def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

df = load_pickle(file_paths["df"])
pipeline = load_pickle(file_paths["pipeline"])

if df is None or pipeline is None:
    st.stop() 

st.header('Enter your inputs')

# property_type
property_type = st.selectbox('Property Type',['flat','house'])

# sector
sector = st.selectbox('Sector',sorted(df['sector'].unique().tolist()))

bedrooms = float(st.selectbox('Number of Bedroom',sorted(df['bedRoom'].unique().tolist())))

bathroom = float(st.selectbox('Number of Bathrooms',sorted(df['bathroom'].unique().tolist())))

balcony = st.selectbox('Balconies',sorted(df['balcony'].unique().tolist()))

property_age = st.selectbox('Property Age',sorted(df['agePossession'].unique().tolist()))

built_up_area = float(st.number_input('Built Up Area'))

servant_room = float(st.selectbox('Servant Room',[0.0, 1.0]))
store_room = float(st.selectbox('Store Room',[0.0, 1.0]))

furnishing_type = st.selectbox('Furnishing Type',sorted(df['furnishing_type'].unique().tolist()))
luxury_category = st.selectbox('Luxury Category',sorted(df['luxury_category'].unique().tolist()))
floor_category = st.selectbox('Floor Category',sorted(df['floor_category'].unique().tolist()))

if st.button('Predict'):

    # form a dataframe
    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']

    # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)

    #st.dataframe(one_df)

    # predict
    base_price = np.expm1(pipeline.predict(one_df))[0]
    low = base_price - 0.22
    high = base_price + 0.22

    # display
    st.text("The price of the flat is between {} Cr and {} Cr".format(round(low,2),round(high,2)))



# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np

# st.set_page_config(page_title="Analysis Model")

# file_paths = {
#     "df": "/mount/src/real-state-model/models/df.pkl",
#     "pipeline": "/mount/src/real-state-model/models/pipeline.pkl"
# }

# def load_pickle(file_path):
#     try:
#         with open(file_path, 'rb') as file:
#             return pickle.load(file)
#     except Exception as e:
#         st.error(f"Error loading {file_path}: {e}")
#         return None

# df = load_pickle(file_paths["df"])
# pipeline = load_pickle(file_paths["pipeline"])

# if df is None or pipeline is None:
#     st.stop() 

# st.header('Enter your inputs')

# # property_type
# property_type = st.selectbox('Property Type', sorted(df['property_type'].unique().tolist()))

# # sector
# sector = st.selectbox('Sector', sorted(df['sector'].unique().tolist()))

# bedrooms = float(st.selectbox('Number of Bedroom', sorted(df['bedRoom'].unique().tolist())))

# bathroom = float(st.selectbox('Number of Bathrooms', sorted(df['bathroom'].unique().tolist())))

# balcony = st.selectbox('Balconies', sorted(df['balcony'].unique().tolist()))

# property_age = st.selectbox('Property Age', sorted(df['agePossession'].unique().tolist()))

# built_up_area = float(st.number_input('Built Up Area'))

# servant_room = float(st.selectbox('Servant Room', [0.0, 1.0]))
# store_room = float(st.selectbox('Store Room', [0.0, 1.0]))

# furnishing_type = st.selectbox('Furnishing Type', sorted(df['furnishing_type'].unique().tolist()))
# luxury_category = st.selectbox('Luxury Category', sorted(df['luxury_category'].unique().tolist()))
# floor_category = st.selectbox('Floor Category', sorted(df['floor_category'].unique().tolist()))

# if st.button('Predict'):

#     # form a dataframe
#     data = [[property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category]]
#     columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
#                'agePossession', 'built_up_area', 'servant room', 'store room',
#                'furnishing_type', 'luxury_category', 'floor_category']

#     # Convert to DataFrame
#     one_df = pd.DataFrame(data, columns=columns)

#     # Check the dataframe columns and types
#     st.write("Input DataFrame Columns and Types:")
#     st.write(one_df.dtypes)
    
#     # Check the model's expected columns
#     st.write("Expected Columns:")
#     st.write(pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out())
    
#     # Predict
#     try:
#         base_price = np.expm1(pipeline.predict(one_df))[0]
#         low = base_price - 0.22
#         high = base_price + 0.22

#         # Display results
#         st.text("The price of the flat is between {} Cr and {} Cr".format(round(low, 2), round(high, 2)))
#     except Exception as e:
#         st.error(f"Error during prediction: {e}")
