import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import re

nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", '', text)
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return ' '.join(tokens)

def preprocess(input_path='data/raw_emails.csv', output_path='data/processed_emails.csv'):
    df = pd.read_csv(input_path)
    df['clean_body'] = df['body'].fillna('').apply(clean_text)
    # Placeholder: if you have labels, merge them here
    # df['label'] = ...
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['clean_body'])
    pd.to_pickle(vectorizer, 'data/vectorizer.pkl')
    feature_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    feature_df['label'] = df.get('label')
    feature_df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

if __name__ == '__main__':
    preprocess()