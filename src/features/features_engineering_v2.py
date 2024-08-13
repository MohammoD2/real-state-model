import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ast

def load_data(file_path):
    """Load dataset from a given path."""
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    return df

def feature_engineering(df):
    """Perform feature engineering on the dataset."""
    
    # Function to extract Super Built up area
    def get_super_built_up_area(text):
        match = re.search(r'Super Built up area (\d+\.?\d*)', text)
        return float(match.group(1)) if match else None

    # Function to extract Built Up area or Carpet area
    def get_area(text, area_type):
        match = re.search(area_type + r'\s*:\s*(\d+\.?\d*)', text)
        return float(match.group(1)) if match else None

    # Function to check if the area is provided in sq.m. and convert it to sqft
    def convert_to_sqft(text, area_value):
        if area_value is None:
            return None
        match = re.search(r'{} \((\d+\.?\d*) sq.m.\)'.format(area_value), text)
        if match:
            sq_m_value = float(match.group(1))
            return sq_m_value * 10.7639  # conversion factor from sq.m. to sqft
        return area_value

    # Extract Super Built up area and convert to sqft if needed
    df['super_built_up_area'] = df['areaWithType'].apply(get_super_built_up_area)
    df['super_built_up_area'] = df.apply(lambda x: convert_to_sqft(x['areaWithType'], x['super_built_up_area']), axis=1)

    # Extract Built Up area and convert to sqft if needed
    df['built_up_area'] = df['areaWithType'].apply(lambda x: get_area(x, 'Built Up area'))
    df['built_up_area'] = df.apply(lambda x: convert_to_sqft(x['areaWithType'], x['built_up_area']), axis=1)

    # Extract Carpet area and convert to sqft if needed
    df['carpet_area'] = df['areaWithType'].apply(lambda x: get_area(x, 'Carpet area'))
    df['carpet_area'] = df.apply(lambda x: convert_to_sqft(x['areaWithType'], x['carpet_area']), axis=1)

    # Handle missing areas for plots
    def extract_plot_area(area_with_type):
        match = re.search(r'Plot area (\d+\.?\d*)', area_with_type)
        return float(match.group(1)) if match else None

    all_nan_df = df[((df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull()))][['price','property_type','area','areaWithType','super_built_up_area','built_up_area','carpet_area']]
    all_nan_df['built_up_area'] = all_nan_df['areaWithType'].apply(extract_plot_area)

    def convert_scale(row):
        if np.isnan(row['area']) or np.isnan(row['built_up_area']):
            return row['built_up_area']
        else:
            if round(row['area']/row['built_up_area']) == 9.0:
                return row['built_up_area'] * 9
            elif round(row['area']/row['built_up_area']) == 11.0:
                return row['built_up_area'] * 10.7
            else:
                return row['built_up_area']

    all_nan_df['built_up_area'] = all_nan_df.apply(convert_scale, axis=1)
    df.update(all_nan_df)

    # Additional Room Feature Engineering
    new_cols = ['study room', 'servant room', 'store room', 'pooja room', 'others']
    for col in new_cols:
        df[col] = df['additionalRoom'].str.contains(col).astype(int)

    # Age Possession Feature Engineering
    def categorize_age_possession(value):
        if pd.isna(value):
            return "Undefined"
        if "0 to 1 Year Old" in value or "Within 6 months" in value or "Within 3 months" in value:
            return "New Property"
        if "1 to 5 Year Old" in value:
            return "Relatively New"
        if "5 to 10 Year Old" in value:
            return "Moderately Old"
        if "10+ Year Old" in value:
            return "Old Property"
        if "Under Construction" in value or "By" in value:
            return "Under Construction"
        try:
            int(value.split(" ")[-1])
            return "Under Construction"
        except:
            return "Undefined"

    df['agePossession'] = df['agePossession'].apply(categorize_age_possession)

    # Furnishing Details Feature Engineering
    def get_furnishing_count(details, furnishing):
        if isinstance(details, str):
            if f"No {furnishing}" in details:
                return 0
            pattern = re.compile(f"(\d+) {furnishing}")
            match = pattern.search(details)
            if match:
                return int(match.group(1))
            elif furnishing in details:
                return 1
        return 0

    all_furnishings = []
    for detail in df['furnishDetails'].dropna():
        furnishings = detail.replace('[', '').replace(']', '').replace("'", "").split(', ')
        all_furnishings.extend(furnishings)
    unique_furnishings = list(set(all_furnishings))

    columns_to_include = [re.sub(r'No |\d+', '', furnishing).strip() for furnishing in unique_furnishings]
    columns_to_include = list(set(columns_to_include))
    columns_to_include = [furnishing for furnishing in columns_to_include if furnishing]

    for furnishing in columns_to_include:
        df[furnishing] = df['furnishDetails'].apply(lambda x: get_furnishing_count(x, furnishing))

    furnishings_df = df[columns_to_include]

    # Furnishing Type Clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(furnishings_df)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(scaled_data)

    df['furnishing_type'] = kmeans.predict(scaled_data)

    # Features Engineering
    df['features'].fillna('', inplace=True)
    df['features_list'] = df['features'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) and x.startswith('[') else [])

    mlb = MultiLabelBinarizer()
    features_binary_matrix = mlb.fit_transform(df['features_list'])

    features_binary_df = pd.DataFrame(features_binary_matrix, columns=mlb.classes_)

    weights = {
        '24/7 Power Backup': 8, '24/7 Water Supply': 4, '24x7 Security': 7, 'ATM': 4, 'Aerobics Centre': 6,
        'Airy Rooms': 8, 'Amphitheatre': 7, 'Badminton Court': 7, 'Banquet Hall': 8, 'Bar/Chill-Out Lounge': 9,
        'Barbecue': 7, 'Basketball Court': 7, 'Billiards': 7, 'Bowling Alley': 8, 'Business Lounge': 9,
        'CCTV Camera Security': 8, 'Cafeteria': 6, 'Car Parking': 6, 'Card Room': 6, 'Centrally Air Conditioned': 9,
        'Changing Area': 6, "Children's Play Area": 7, 'Cigar Lounge': 9, 'Clinic': 5, 'Club House': 9,
        'Concierge Service': 9, 'Conference room': 8, 'Creche/Day care': 7, 'Cricket Pitch': 7, 'Doctor on Call': 6,
        'Earthquake Resistant': 5, 'Entrance Lobby': 7, 'False Ceiling Lighting': 6, 'Feng Shui / Vaastu Compliant': 5,
        'Fire Fighting Systems': 8, 'Fitness Centre / GYM': 8, 'Flower Garden': 7, 'Food Court': 6, 'Foosball': 5,
        'Football': 7, 'Fountain': 7, 'Gated Community': 7, 'Golf Course': 10, 'Grocery Shop': 6, 'Gymnasium': 8,
        'High Ceiling Height': 8, 'High Speed Elevators': 8, 'Infinity Pool': 9, 'Intercom Facility': 7,
        'Internal Street Lights': 6, 'Internet/wi-fi connectivity': 7, 'Jacuzzi': 9, 'Jogging Track': 7,
        'Landscape Garden': 8, 'Laundry': 6, 'Lawn Tennis Court': 8, 'Library': 8, 'Lounge': 8, 'Low Density Society': 7,
        'Maintenance Staff': 7, 'Meditation Area': 6, 'Multipurpose Court': 8, 'Multipurpose Hall': 8, 'Park': 8,
        'Party Lawn': 8, 'Paved Compound': 6, 'Pergola': 6, 'Piped Gas': 7, 'Pool Table': 7, 'Power Back up Lift': 8,
        'Property Staff': 7, 'Rain Water Harvesting': 8, 'Reading Lounge': 7, 'Reflexology Park': 7, 'Reserved Parking': 8,
        'Restaurant': 7, 'RO System': 7, 'Sauna': 9, 'School': 6, 'Senior Citizen Sitout': 7, 'Sewage Treatment Plant': 7,
        'Shopping Centre': 7, 'Skating Rink': 6, 'Solar Lighting': 7, 'Solar Water Heating': 7, 'Spa': 9,
        'Squash Court': 8, 'Steam Room': 9, 'Street Lighting': 7, 'Sun Deck': 8, 'Swimming Pool': 9, 'Table Tennis': 7,
        'Theatre': 8, 'Toddler Pool': 7, 'Valet Parking': 9, 'Video Door Security': 8, 'Waiting Lounge': 7,
        'Water Softener Plant': 7, 'Wi-Fi Connectivity': 7, 'Yoga/Meditation Area': 7
    }

    df['features'] = df['features'].fillna('')
    
    # Create binary features list from 'features'
    df['features_list'] = df['features'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) and x.startswith('[') else [])

    mlb = MultiLabelBinarizer()
    features_binary_matrix = mlb.fit_transform(df['features_list'])

    features_binary_df = pd.DataFrame(features_binary_matrix, columns=mlb.classes_)

    # Ensure all features from weights are in the DataFrame
    for feature in weights:
        if feature not in features_binary_df.columns:
            features_binary_df[feature] = 0

    # Calculate luxury score
    def calculate_luxury_score(row):
        total_score = sum([weights[feature] for feature in weights if row[feature] == 1])
        return total_score

    df['luxury_score'] = features_binary_df.apply(calculate_luxury_score, axis=1)
    
    # ... (additional processing remains unchanged)

    return df

def save_data(df, output_path):
    """Save the processed DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)

# Example usage
file_path = 'E:\Work files\Real_state\data\processed\gurgaon_properties_cleaned_v1.csv'
output_path = 'E:\Work files\Real_state\data\processed\gurgaon_properties_cleaned_v2.csv'

df = load_data(file_path)
df = feature_engineering(df)
save_data(df, output_path)
