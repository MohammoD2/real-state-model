import pandas as pd
import pickle
import numpy as np

def load_model():
    with open('E:\Work files\Real_state\models\pipeline.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_input(data):
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']
    df = pd.DataFrame(data, columns=columns)
    return df

def make_prediction(model, input_data):
    preprocessed_data = preprocess_input(input_data)
    predictions = model.predict(preprocessed_data)
    return np.expm1(predictions)

def main():
    model = load_model()
    data = [['house', 'sector 102', 4, 3, '3+', 'New Property', 2750, 0, 0, 'unfurnished', 'Low', 'Low Floor']]
    predictions = make_prediction(model, data)
    print(f"Predictions: {predictions}")

if __name__ == "__main__":
    main()
