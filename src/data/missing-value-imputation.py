import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set global display options
pd.set_option('display.max_columns', None)

def load_data(filepath):
    """Load dataset and return DataFrame."""
    df = pd.read_csv(filepath)
    return df

def check_missing_values(df):
    """Check for missing values in the DataFrame."""
    return df.isnull().sum()

def fill_built_up_area(df):
    """Fill missing built_up_area based on related features."""
    # Cases where super_built_up_area and carpet_area are present
    sbc_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]
    sbc_df['built_up_area'].fillna(round(((sbc_df['super_built_up_area']/1.105) + (sbc_df['carpet_area']/0.9))/2), inplace=True)
    
    # Cases where only super_built_up_area is present
    sb_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull())]
    sb_df['built_up_area'].fillna(round(sb_df['super_built_up_area']/1.105), inplace=True)
    
    # Cases where only carpet_area is present
    c_df = df[(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]
    c_df['built_up_area'].fillna(round(c_df['carpet_area']/0.9), inplace=True)
    
    df.update(sbc_df)
    df.update(sb_df)
    df.update(c_df)
    return df

def handle_anomalies(df):
    """Fix anomalies in built_up_area and drop unnecessary columns."""
    anamoly_df = df[(df['built_up_area'] < 2000) & (df['price'] > 2.5)][['price','area','built_up_area']]
    anamoly_df['built_up_area'] = anamoly_df['area']
    df.update(anamoly_df)
    df.drop(columns=['area','areaWithType','super_built_up_area','carpet_area','area_room_ratio'], inplace=True)
    return df

def impute_floor_num(df):
    """Impute missing floorNum based on property type."""
    median_floor_num = df[df['property_type'] == 'house']['floorNum'].median()
    df['floorNum'].fillna(median_floor_num, inplace=True)
    return df

def drop_facing_column(df):
    """Drop 'facing' column."""
    df.drop(columns=['facing'], inplace=True)
    return df

def impute_age_possession(df):
    """Impute missing agePossession using mode-based imputation."""
    def mode_based_imputation(row):
        if row['agePossession'] == 'Undefined':
            mode_value = df[(df['sector'] == row['sector']) & (df['property_type'] == row['property_type'])]['agePossession'].mode()
            return mode_value.iloc[0] if not mode_value.empty else np.nan
        else:
            return row['agePossession']

    df['agePossession'] = df.apply(mode_based_imputation, axis=1)
    return df

def save_cleaned_data(df, filepath):
    """Save the cleaned DataFrame to a CSV file."""
    df.to_csv(filepath, index=False)

# Main execution
if __name__ == "__main__":
    df = load_data('E:\Work files\Real_state\data\processed\gurgaon_properties_outlier_treated.csv')

    # Initial checks
    print(check_missing_values(df))

    # Data cleaning and imputation
    df = fill_built_up_area(df)
    df = handle_anomalies(df)
    df = impute_floor_num(df)
    df = drop_facing_column(df)
    df = impute_age_possession(df)

    # Check final missing values and save cleaned data
    print(check_missing_values(df))
    save_cleaned_data(df, 'E:\Work files\Real_state\data\processed\gurgaon_properties_missing_value_imputation.csv')
