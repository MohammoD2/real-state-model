import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import category_encoders as ce

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df['furnishing_type'] = df['furnishing_type'].replace({0.0:'unfurnished',1.0:'semifurnished',2.0:'furnished'})
    X = df.drop(columns=['price'])
    y = np.log1p(df['price'])
    return X, y

def create_preprocessor():
    columns_to_encode = ['property_type', 'sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
            ('cat', OrdinalEncoder(), columns_to_encode),
            ('cat1', OneHotEncoder(drop='first', sparse_output=False), ['sector', 'agePossession'])
        ],
        remainder='passthrough'
    )
    return preprocessor

def train_and_save_model(X, y):
    preprocessor = create_preprocessor()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])

    param_grid = {
        'regressor__n_estimators': [50, 100, 200, 300],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__max_samples': [0.1, 0.25, 0.5, 1.0],
        'regressor__max_features': ['auto', 'sqrt']
    }

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    search = GridSearchCV(pipeline, param_grid, cv=kfold, scoring='r2', n_jobs=-1, verbose=4)
    search.fit(X, y)

    final_pipe = search.best_estimator_
    
    with open('E:\Work files\Real_state\models\pipeline.pkl', 'wb') as file:
        pickle.dump(final_pipe, file)
    print("Model saved as 'pipeline.pkl'")

def main():
    df = load_data("E:\Work files\Real_state\data\processed\gurgaon_properties_post_feature_selection.csv")
    X, y = preprocess_data(df)
    train_and_save_model(X, y)

if __name__ == "__main__":
    main()
