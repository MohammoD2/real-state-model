import pandas as pd
import numpy as np
from utils import load_data, plot_distribution, plot_boxplot, plot_categorical_distribution, plot_pie_chart, plot_ecdf, handle_missing_values

def main(filepath):
    # Load data
    df = load_data(filepath)
    
    # Analysis on property type
    plot_categorical_distribution(df, 'property_type', 'Property Type Distribution')
    
    # Society Analysis
    print(f"Unique societies: {df['society'].nunique()}")
    plot_categorical_distribution(df[df['society'] != 'independent'], 'society', 'Top Societies')
    
    # Sector Analysis
    plot_categorical_distribution(df, 'sector', 'Top Sectors')
    
    # Price Analysis
    handle_missing_values(df, 'price')
    plot_distribution(df, 'price', 'Price Distribution')
    plot_boxplot(df, 'price', 'Price Boxplot')
    
    # Price per Sqft Analysis
    handle_missing_values(df, 'price_per_sqft')
    plot_distribution(df, 'price_per_sqft', 'Price per Sqft Distribution')
    plot_boxplot(df, 'price_per_sqft', 'Price per Sqft Boxplot')
    
    # Bedrooms Analysis
    plot_categorical_distribution(df, 'bedRoom', 'Bedrooms Distribution')
    
    # Bathrooms Analysis
    plot_categorical_distribution(df, 'bathroom', 'Bathrooms Distribution')
    
    # Balcony Analysis
    plot_categorical_distribution(df, 'balcony', 'Balcony Distribution')
    
    # Floor Number Analysis
    plot_distribution(df, 'floorNum', 'Floor Number Distribution')
    plot_boxplot(df, 'floorNum', 'Floor Number Boxplot')
    
    # Facing Analysis
    df = handle_missing_values(df, 'facing', strategy='mode')
    plot_categorical_distribution(df, 'facing', 'Facing Distribution')
    
    # Age of Possession Analysis
    plot_categorical_distribution(df, 'agePossession', 'Age of Possession Distribution')
    
    # Super Built-Up Area Analysis
    plot_distribution(df, 'super_built_up_area', 'Super Built-Up Area Distribution')
    plot_boxplot(df, 'super_built_up_area', 'Super Built-Up Area Boxplot')
    
    # Built-Up Area Analysis
    plot_distribution(df, 'built_up_area', 'Built-Up Area Distribution')
    plot_boxplot(df, 'built_up_area', 'Built-Up Area Boxplot')
    
    # Carpet Area Analysis
    plot_distribution(df, 'carpet_area', 'Carpet Area Distribution')
    plot_boxplot(df, 'carpet_area', 'Carpet Area Boxplot')
    
    # Additional Rooms Analysis
    additional_rooms = ['study room', 'servant room', 'store room', 'pooja room', 'others']
    for room in additional_rooms:
        plot_pie_chart(df, room, f'Distribution of {room.title()}')
    
    # Furnishing Type Analysis
    plot_pie_chart(df, 'furnishing_type', 'Furnishing Type Distribution')
    
    # Luxury Score Analysis
    handle_missing_values(df, 'luxury_score')
    plot_distribution(df, 'luxury_score', 'Luxury Score Distribution')
    plot_boxplot(df, 'luxury_score', 'Luxury Score Boxplot')

if __name__ == "__main__":
    main('E:\Work files\Real_state\data\processed\gurgaon_properties_cleaned_v2.csv')
