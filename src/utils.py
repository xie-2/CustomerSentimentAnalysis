# Common functions: loading/saving models, a single process_text() wrapper, email‐parsing helpers.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    """
    Load and preprocess customer experience data.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Preprocessed DataFrame
    """
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df = df.fillna({
        'Age': df['Age'].median(),
        'Num_Interactions': 0,
        'Feedback_Score': df['Feedback_Score'].median(),
        'Products_Purchased': 0,
        'Products_Viewed': 0,
        'Time_Spent_on_Site': 0,
        'Satisfaction_Score': df['Satisfaction_Score'].median()
    })
    
    return df

def scale_features(df, feature_columns):
    """
    Scale numerical features using StandardScaler.
    
    Args:
        df: DataFrame with features
        feature_columns: List of column names to scale
        
    Returns:
        Scaled DataFrame
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df_scaled

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance from the model.
    
    Args:
        model: Trained model with coef_ attribute
        feature_names: List of feature names
    """
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': abs(model.coef_[0])
    })
    importance = importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance, x='Importance', y='Feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('data/feature_importance.png')
    plt.close()

def plot_correlation_matrix(df, feature_columns):
    """
    Plot correlation matrix for numerical features.
    
    Args:
        df: DataFrame with features
        feature_columns: List of numerical column names
    """
    corr = df[feature_columns].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('data/correlation_matrix.png')
    plt.close()

def analyze_customer_segments(df, feature_columns):
    """
    Analyze customer segments based on key metrics.
    
    Args:
        df: DataFrame with customer data
        feature_columns: List of feature column names
    """
    # Calculate segment statistics
    segments = df.groupby('Retention_Status').agg({
        'Age': 'mean',
        'Num_Interactions': 'mean',
        'Feedback_Score': 'mean',
        'Products_Purchased': 'mean',
        'Time_Spent_on_Site': 'mean',
        'Satisfaction_Score': 'mean'
    }).round(2)
    
    # Save segment analysis
    segments.to_csv('data/customer_segments.csv')
    
    # Plot segment comparisons
    plt.figure(figsize=(15, 8))
    segments.plot(kind='bar')
    plt.title('Customer Segment Analysis')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/segment_analysis.png')
    plt.close()
    
    return segments