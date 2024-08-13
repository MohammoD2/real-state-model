import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Function to load data
def load_data(filepath):
    df = pd.read_csv(filepath).drop_duplicates()
    return df

# Function to plot property type vs price
def plot_property_type_vs_price(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=df['property_type'], y=df['price'], estimator=np.median)
    plt.title('Property Type vs Median Price')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df['property_type'], y=df['price'])
    plt.title('Property Type vs Price Distribution')
    plt.show()

# Function to plot property type vs area
def plot_property_type_vs_area(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=df['property_type'], y=df['built_up_area'], estimator=np.median)
    plt.title('Property Type vs Median Built-up Area')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df['property_type'], y=df['built_up_area'])
    plt.title('Property Type vs Built-up Area Distribution')
    plt.show()

# Function to remove outliers
def remove_outliers(df):
    df = df[df['built_up_area'] != 737147]
    return df

# Function to plot property type vs price per sqft
def plot_property_type_vs_price_per_sqft(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=df['property_type'], y=df['price_per_sqft'], estimator=np.median)
    plt.title('Property Type vs Median Price per sqft')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df['property_type'], y=df['price_per_sqft'])
    plt.title('Property Type vs Price per sqft Distribution')
    plt.show()

# Function to plot sector analysis
def plot_sector_analysis(df):
    # Group by 'sector' and calculate the average price
    avg_price_per_sector = df.groupby('sector')['price'].mean().reset_index()

    # Function to extract sector numbers
    def extract_sector_number(sector_name):
        match = re.search(r'\d+', sector_name)
        if match:
            return int(match.group())
        else:
            return float('inf')  # Return a large number for non-numbered sectors

    avg_price_per_sector['sector_number'] = avg_price_per_sector['sector'].apply(extract_sector_number)

    # Sort by sector number
    avg_price_per_sector_sorted_by_sector = avg_price_per_sector.sort_values(by='sector_number')

    # Plot the heatmap
    plt.figure(figsize=(5, 25))
    sns.heatmap(avg_price_per_sector_sorted_by_sector.set_index('sector')[['price']], annot=True, fmt=".2f", linewidths=.5)
    plt.title('Average Price per Sector (Sorted by Sector Number)')
    plt.xlabel('Average Price')
    plt.ylabel('Sector')
    plt.show()

    avg_price_per_sqft_sector = df.groupby('sector')['price_per_sqft'].mean().reset_index()
    avg_price_per_sqft_sector['sector_number'] = avg_price_per_sqft_sector['sector'].apply(extract_sector_number)
    avg_price_per_sqft_sector_sorted_by_sector = avg_price_per_sqft_sector.sort_values(by='sector_number')

    # Plot the heatmap
    plt.figure(figsize=(5, 25))
    sns.heatmap(avg_price_per_sqft_sector_sorted_by_sector.set_index('sector')[['price_per_sqft']], annot=True, fmt=".2f", linewidths=.5)
    plt.title('Average Price per sqft per Sector (Sorted by Sector Number)')
    plt.xlabel('Average Price per sqft')
    plt.ylabel('Sector')
    plt.show()

    luxury_score = df.groupby('sector')['luxury_score'].mean().reset_index()
    luxury_score['sector_number'] = luxury_score['sector'].apply(extract_sector_number)
    luxury_score_sector = luxury_score.sort_values(by='sector_number')

    # Plot the heatmap
    plt.figure(figsize=(5, 25))
    sns.heatmap(luxury_score_sector.set_index('sector')[['luxury_score']], annot=True, fmt=".2f", linewidths=.5)
    plt.title('Luxury Score per Sector (Sorted by Sector Number)')
    plt.xlabel('Luxury Score')
    plt.ylabel('Sector')
    plt.show()

# Function to plot correlations
def plot_correlation(df):
    plt.figure(figsize=(8, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

# Function to plot scatter plots
def plot_scatter(df):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(df[df['area'] < 10000]['area'], df['price'], hue=df['bedRoom'])
    plt.title('Price vs Area by Number of Bedrooms')
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.scatterplot(df[df['area'] < 10000]['area'], df['price'], hue=df['agePossession'])
    plt.title('Price vs Area by Age of Possession')
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.scatterplot(df[df['area'] < 10000]['area'], df['price'], hue=df['furnishing_type'].astype('category'))
    plt.title('Price vs Area by Furnishing Type')
    plt.show()
if __name__ == "__main__":
    df = load_data("E:\Work files\Real_state\data\processed\gurgaon_properties_cleaned_v2.csv")
    df = remove_outliers(df)
    plot_property_type_vs_price(df)
    plot_property_type_vs_area(df)
    plot_property_type_vs_price_per_sqft(df)
    plot_sector_analysis(df)
    plot_correlation(df)
    plot_scatter(df)
