import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    df = pd.read_csv(filepath)
    df.drop_duplicates(inplace=True)
    return df

def plot_distribution(df, column, title, bins=50):
    plt.figure(figsize=(12, 6))
    sns.histplot(df[column], bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_boxplot(df, column, title):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df[column], color='lightgreen')
    plt.title(title)
    plt.xlabel(column)
    plt.grid()
    plt.show()

def plot_categorical_distribution(df, column, title):
    plt.figure(figsize=(12, 6))
    df[column].value_counts().plot(kind='bar')
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()
    
def plot_pie_chart(df, column, title):
    plt.figure(figsize=(8, 8))
    df[column].value_counts().plot(kind='pie', autopct='%0.2f%%')
    plt.title(title)
    plt.ylabel('')
    plt.show()

def plot_ecdf(df, column):
    ecdf = df[column].value_counts().sort_index().cumsum() / len(df[column])
    plt.figure(figsize=(12, 6))
    plt.plot(ecdf.index, ecdf, marker='.', linestyle='none')
    plt.title(f'ECDF of {column}')
    plt.xlabel(column)
    plt.ylabel('Cumulative Proportion')
    plt.grid()
    plt.show()

def handle_missing_values(df, column, strategy='mean'):
    if strategy == 'mean':
        df[column].fillna(df[column].mean(), inplace=True)
    elif strategy == 'median':
        df[column].fillna(df[column].median(), inplace=True)
    elif strategy == 'mode':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        raise ValueError("Strategy not recognized. Use 'mean', 'median', or 'mode'.")
    return df
