# Load data/processed_emails.csv.
# Split into train/test sets (e.g. 80/20).
# Fit your Naive Bayes classifier on the training data.
# Evaluate on the test set (accuracy, precision/recall, confusion matrix).
# Serialize (pickle) both the trained model and the vectorizer into models/nb_model.pkl (and models/vectorizer.pkl, if you haven't already).
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os


def train_model(input_path='data/customer_experience_data.csv'):
    # Load the data
    df = pd.read_csv(input_path)
    
    # Select features for prediction
    feature_columns = [
        'Age', 'Gender_Encoded', 'Location_Encoded', 
        'Num_Interactions', 'Feedback_Score', 
        'Products_Purchased', 'Products_Viewed', 
        'Time_Spent_on_Site', 'Satisfaction_Score'
    ]
    
    # Prepare features and target
    X = df[feature_columns]
    y = df['Retention_Status_Encoded']  # Predicting retention status
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': abs(model.coef_[0])
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values('Importance', ascending=False))
    
    # Save the model
    os.makedirs('data', exist_ok=True)
    joblib.dump(model, 'data/retention_model.joblib')
    print("\nModel saved to data/retention_model.joblib")


if __name__ == '__main__':
    train_model()