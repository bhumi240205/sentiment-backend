from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Initialize FastAPI app
app = FastAPI(title="Sentiment Analysis API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('lemmatizers/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Auto-Train Model if not exists
model_path = 'sentiment_model.pkl'
if not os.path.exists(model_path):
    print(" Model not found! Training on Render...")
    
    # Load data (LIMIT TO 1000 ROWS FOR MEMORY)
    df = pd.read_csv('data.csv').sample(n=1000, random_state=42)
    
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
    print(" Model trained and saved!")
else:
    print(" Model already exists. Loading...")
    model = joblib.load(model_path)

# Define the cleaning function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Define Input/Output structure
class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    scores: dict

# Create the Prediction Endpoint
@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: TextRequest):
    try:
        clean_input = clean_text(request.text)
        
        if not clean_input:
            raise HTTPException(status_code=400, detail="Text is empty after cleaning.")

        prediction = model.predict([clean_input])[0]
        probabilities = model.predict_proba([clean_input])[0]

        label_map = {0: "Negative", 1: "Positive"}
        final_label = label_map[prediction]

        scores = {
            "Negative": float(probabilities[0]),
            "Positive": float(probabilities[1])
        }

        return {
            "label": final_label,
            "scores": scores
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))