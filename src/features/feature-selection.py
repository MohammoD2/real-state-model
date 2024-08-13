import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score

# Function to load data
def load_data(filepath):
    """Load dataset and return DataFrame."""
    return pd.read_csv(filepath)

# Function to preprocess data
def preprocess_data(df):
    """Perform initial data preprocessing."""
    train_df = df.drop(columns=['society', 'price_per_sqft'])
    return train_df

# Function to categorize luxury scores
def categorize_luxury(score):
    if 0 <= score < 50:
        return "Low"
    elif 50 <= score < 150:
        return "Medium"
    elif 150 <= score <= 175:
        return "High"
    else:
        return None

# Function to categorize floor numbers
def categorize_floor(floor):
    if 0 <= floor <= 2:
        return "Low Floor"
    elif 3 <= floor <= 10:
        return "Mid Floor"
    elif 11 <= floor <= 51:
        return "High Floor"
    else:
        return None

# Function to encode categorical features
def encode_categorical_features(df):
    """Label encode categorical features."""
    data_label_encoded = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        oe = OrdinalEncoder()
        data_label_encoded[col] = oe.fit_transform(data_label_encoded[[col]])
        print(f"Categories for {col}: {oe.categories_}")
    
    return data_label_encoded

# Function to perform feature importance techniques
def feature_importance_analysis(X, y):
    """Analyze feature importance using multiple techniques."""
    # Technique 1: Correlation Analysis
    X_with_price = X.copy()
    X_with_price['price'] = y
    corr = X_with_price.corr()['price'].iloc[:-1].to_frame().reset_index().rename(columns={'index': 'feature', 'price': 'corr_coeff'})
    
    # Technique 2: Random Forest Feature Importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'rf_importance': rf.feature_importances_
    }).sort_values(by='rf_importance', ascending=False)
    
    # Technique 3: Gradient Boosting Feature Importance
    gb = GradientBoostingRegressor()
    gb.fit(X, y)
    gb_importance = pd.DataFrame({
        'feature': X.columns,
        'gb_importance': gb.feature_importances_
    }).sort_values(by='gb_importance', ascending=False)
    
    # Technique 4: Permutation Importance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf.fit(X_train, y_train)
    perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=42)
    perm_importance_df = pd.DataFrame({
        'feature': X.columns,
        'permutation_importance': perm_importance.importances_mean
    }).sort_values(by='permutation_importance', ascending=False)
    
    # Technique 5: LASSO
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X_scaled, y)
    lasso_importance = pd.DataFrame({
        'feature': X.columns,
        'lasso_coeff': lasso.coef_
    }).sort_values(by='lasso_coeff', ascending=False)
    
    # Technique 6: RFE
    estimator = RandomForestRegressor()
    selector = RFE(estimator, n_features_to_select=X.shape[1], step=1)
    selector = selector.fit(X, y)
    selected_features = X.columns[selector.support_]
    rfe_importance = pd.DataFrame({
        'feature': selected_features,
        'rfe_score': selector.estimator_.feature_importances_
    }).sort_values(by='rfe_score', ascending=False)
    
    # Technique 7: Linear Regression Weights
    lin_reg = LinearRegression()
    lin_reg.fit(X_scaled, y)
    lin_reg_importance = pd.DataFrame({
        'feature': X.columns,
        'reg_coeffs': lin_reg.coef_
    }).sort_values(by='reg_coeffs', ascending=False)
    
    # Technique 8: SHAP
    import shap
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)   
    shap_importance = pd.DataFrame({
        'feature': X.columns,
        'SHAP_score': np.abs(shap_values).mean(axis=0)
    }).sort_values(by='SHAP_score', ascending=False)
    
    # Combine all feature importance results
    final_fi_df = corr.merge(rf_importance, on='feature')\
                      .merge(gb_importance, on='feature')\
                      .merge(perm_importance_df, on='feature')\
                      .merge(lasso_importance, on='feature')\
                      .merge(rfe_importance, on='feature')\
                      .merge(lin_reg_importance, on='feature')\
                      .merge(shap_importance, on='feature')\
                      .set_index('feature')
    
    return final_fi_df

# Function to evaluate model performance
def evaluate_model(X, y):
    """Evaluate model performance using cross-validation."""
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    return scores.mean()

# Function to save the processed data
def save_processed_data(df, filepath):
    """Save the processed DataFrame to a CSV file."""
    df.to_csv(filepath, index=False)

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    df = load_data('E:\Work files\Real_state\data\processed\gurgaon_properties_missing_value_imputation.csv')
    train_df = preprocess_data(df)

    # Add new features
    train_df['luxury_category'] = train_df['luxury_score'].apply(categorize_luxury)
    train_df['floor_category'] = train_df['floorNum'].apply(categorize_floor)
    train_df.drop(columns=['floorNum', 'luxury_score'], inplace=True)

    # Encode categorical features
    data_label_encoded = encode_categorical_features(train_df)

    # Split data
    X = data_label_encoded.drop('price', axis=1)
    y = data_label_encoded['price']

    # Feature importance analysis
    fi_df = feature_importance_analysis(X, y)
    print(fi_df)

    # Evaluate model performance
    mean_r2_score = evaluate_model(X, y)
    print(f"Mean R2 Score: {mean_r2_score}")

    # Save the processed data
    export_df = X.copy()
    export_df['price'] = y
    export_df = export_df.drop(columns=['pooja room', 'study room', 'others'])
    save_processed_data(export_df, 'E:\Work files\Real_state\data\processed\gurgaon_properties_post_feature_selection.csv')
