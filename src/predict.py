# Load your nb_model.pkl and vectorizer.pkl.
# Accept raw email text (from file, stdin, or direct function call).
# Preprocess it (reuse your cleaning pipeline).
# Vectorize and predict sentiment, then print/log the result.
import argparse
import joblib
import pandas as pd


def predict_sentiment(text, model_path='data/sentiment_model.joblib', vectorizer_path='data/vectorizer.joblib'):
    # Load the model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # Transform the input text
    text_vec = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]
    
    return {
        'sentiment': prediction,
        'confidence': max(probabilities)
    }

def predict_retention(customer_data, model_path='data/retention_model.joblib'):
    """
    Predict retention status for a customer.
    
    Args:
        customer_data: Dictionary containing customer features:
            - Age
            - Gender_Encoded
            - Location_Encoded
            - Num_Interactions
            - Feedback_Score
            - Products_Purchased
            - Products_Viewed
            - Time_Spent_on_Site
            - Satisfaction_Score
    """
    # Load the model
    model = joblib.load(model_path)
    
    # Convert input to DataFrame
    df = pd.DataFrame([customer_data])
    
    # Make prediction
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    
    return {
        'retention_status': prediction,
        'confidence': max(probabilities)
    }

def predict_from_file(input_path, output_path='data/predictions.csv'):
    # Load the data
    df = pd.read_csv(input_path)
    
    # Load the model
    model = joblib.load('data/retention_model.joblib')
    
    # Select features
    feature_columns = [
        'Age', 'Gender_Encoded', 'Location_Encoded', 
        'Num_Interactions', 'Feedback_Score', 
        'Products_Purchased', 'Products_Viewed', 
        'Time_Spent_on_Site', 'Satisfaction_Score'
    ]
    
    # Make predictions
    X = df[feature_columns]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Add predictions to the dataframe
    df['predicted_retention'] = predictions
    df['prediction_confidence'] = probabilities.max(axis=1)
    
    # Save results
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict customer retention')
    parser.add_argument('--file', type=str, help='Path to CSV file with customer data')
    parser.add_argument('--output', type=str, default='data/predictions.csv',
                      help='Path to save predictions (default: data/predictions.csv)')
    
    args = parser.parse_args()
    
    if args.file:
        # Predict from file
        predict_from_file(args.file, args.output)
    else:
        # Example prediction with sample data
        customer = {
            'Age': 30,
            'Gender_Encoded': 1,
            'Location_Encoded': 2,
            'Num_Interactions': 5,
            'Feedback_Score': 4,
            'Products_Purchased': 3,
            'Products_Viewed': 10,
            'Time_Spent_on_Site': 45,
            'Satisfaction_Score': 4
        }
        
        result = predict_retention(customer)
        print(f"\nExample prediction:")
        print(f"Customer data: {customer}")
        print(f"Predicted retention status: {result['retention_status']}")
        print(f"Confidence: {result['confidence']:.2f}")