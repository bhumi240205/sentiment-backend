# --- ADD THIS SECTION (Auto-Training) ---
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Check if model exists, if not, train it
model_path = 'sentiment_model.pkl'
if not os.path.exists(model_path):
    print("⚠️ Model not found! Training on Render...")
    
    # Load data
    df = pd.read_csv('data.csv')
    
    # Clean data
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)
    
    df['clean_text'] = df['text'].apply(clean_text)
    df['sentiment_num'] = df['sentiment'].map({'negative': 0, 'positive': 1})
    df = df.dropna(subset=['sentiment_num'])
    
    # Train model
    X = df['clean_text']
    y = df['sentiment_num']
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, model_path)
    print("✅ Model trained and saved!")
else:
    print("✅ Model already exists. Loading...")
    model = joblib.load(model_path)
# ------------------------------------------