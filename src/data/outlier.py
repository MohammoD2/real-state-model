import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set global display options
pd.set_option('display.max_columns', None)

def load_data(filepath):
    """Load dataset and remove duplicates."""
    df = pd.read_csv(filepath).drop_duplicates()
    return df

def calculate_iqr(df, column):
    """Calculate the IQR for a given column and return outlier bounds."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def detect_outliers(df, column):
    """Detect outliers in a given column using IQR method."""
    lower_bound, upper_bound = calculate_iqr(df, column)
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def clean_outliers(df, column, upper_limit=None):
    """Clean outliers from a column and optionally cap at upper_limit."""
    if upper_limit:
        df = df[df[column] <= upper_limit]
    else:
        outliers, lower_bound, upper_bound = detect_outliers(df, column)
        df = df[~df.index.isin(outliers.index)]
    return df

def visualize_distribution(df, column):
    """Visualize the distribution of a column."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df[column], kde=True, bins=50)
    plt.title(f'Distribution of {column}')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    
    plt.tight_layout()
    plt.show()

def fix_price_per_sqft(df):
    """Fix price_per_sqft for data consistency."""
    outliers_sqft = detect_outliers(df, 'price_per_sqft')[0]
    df.update(outliers_sqft)
    return df

def update_area(df):
    """Correct and clean area related anomalies."""
    df.loc[df['area'] > 100000, 'area'] = np.nan
    df = df[df['area'] < 100000]
    df.loc[2131, 'area'] = 1812
    return df

def clean_feature(df, column, threshold, update_values=None):
    """Clean a feature based on threshold and optionally update values."""
    df = df[df[column] <= threshold]
    if update_values:
        for index, value in update_values.items():
            df.loc[index, column] = value
    return df

def save_cleaned_data(df, filepath):
    """Save the cleaned dataframe to a CSV file."""
    df.to_csv(filepath, index=False)

# Main execution
if __name__ == "__main__":
    df = load_data('E:\Work files\Real_state\data\processed\gurgaon_properties_cleaned_v2.csv')

    # Clean and visualize data
    df = fix_price_per_sqft(df)
    df = update_area(df)
    
    df = clean_feature(df, 'bedRoom', 10)
    df = clean_feature(df, 'bathroom', 10)
    df = clean_feature(df, 'super_built_up_area', 6000)
    df = clean_feature(df, 'built_up_area', 10000)
    df = clean_feature(df, 'carpet_area', 10000, {2131: 1812})
    df = clean_feature(df, 'price_per_sqft', 42000)
    
    # Visualize cleaned features
    for col in ['price', 'price_per_sqft', 'area', 'bedRoom', 'bathroom', 'super_built_up_area', 'built_up_area', 'carpet_area', 'luxury_score']:
        visualize_distribution(df, col)
    
    # Save cleaned data
    save_cleaned_data(df, 'E:\Work files\Real_state\data\processed\gurgaon_properties_cleaned_final.csv')
