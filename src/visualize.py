# Generate a confusion matrix (and optionally ROC curve).
# Plot overall sentiment distribution over a batch of emails.
# Save figures into a reports/ folder (or inline in notebooks, if you prefer).

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_and_preprocess_data, plot_correlation_matrix, analyze_customer_segments

def create_visualizations(input_path='data/customer_experience_data.csv'):
    """
    Create comprehensive visualizations of customer experience data.
    
    Args:
        input_path: Path to the customer experience data CSV file
    """
    # Load and preprocess data
    df = load_and_preprocess_data(input_path)
    
    # Define feature columns
    feature_columns = [
        'Age', 'Num_Interactions', 'Feedback_Score',
        'Products_Purchased', 'Products_Viewed',
        'Time_Spent_on_Site', 'Satisfaction_Score'
    ]
    
    # 1. Distribution of Key Metrics
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(feature_columns, 1):
        plt.subplot(3, 3, i)
        sns.histplot(data=df, x=feature, hue='Retention_Status', multiple="stack")
        plt.title(f'{feature} Distribution')
    plt.tight_layout()
    plt.savefig('data/distributions.png')
    plt.close()
    
    # 2. Correlation Matrix
    plot_correlation_matrix(df, feature_columns)
    
    # 3. Customer Segments Analysis
    analyze_customer_segments(df, feature_columns)
    
    # 4. Gender and Location Analysis
    plt.figure(figsize=(12, 5))
    
    # Gender distribution
    plt.subplot(1, 2, 1)
    gender_retention = pd.crosstab(df['Gender'], df['Retention_Status'], normalize='index') * 100
    gender_retention.plot(kind='bar', stacked=True)
    plt.title('Retention by Gender')
    plt.ylabel('Percentage')
    
    # Location distribution
    plt.subplot(1, 2, 2)
    location_retention = pd.crosstab(df['Location'], df['Retention_Status'], normalize='index') * 100
    location_retention.plot(kind='bar', stacked=True)
    plt.title('Retention by Location')
    plt.ylabel('Percentage')
    
    plt.tight_layout()
    plt.savefig('data/demographic_analysis.png')
    plt.close()
    
    # 5. Interaction Analysis
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='Num_Interactions', y='Satisfaction_Score', 
                   hue='Retention_Status', alpha=0.6)
    plt.title('Satisfaction vs. Number of Interactions')
    plt.savefig('data/interaction_analysis.png')
    plt.close()
    
    # 6. Time Spent Analysis
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Retention_Status', y='Time_Spent_on_Site')
    plt.title('Time Spent on Site by Retention Status')
    plt.savefig('data/time_analysis.png')
    plt.close()
    
    # 7. Product Engagement Analysis
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='Products_Viewed', y='Products_Purchased',
                   hue='Retention_Status', alpha=0.6)
    plt.title('Product Views vs. Purchases')
    plt.savefig('data/product_engagement.png')
    plt.close()
    
    print("All visualizations have been saved to the data/ directory")

if __name__ == '__main__':
    create_visualizations()