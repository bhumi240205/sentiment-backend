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
import warnings

warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(title="Syncaura Sentiment Analysis API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('lemmatizers/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Configuration
model_path = 'sentiment_model.pkl'
data_3class_path = 'data_3class.csv'
min_data_rows = 100  # If data_3class.csv has > this many rows, force retrain

# Auto-Train Model if needed
def train_model():
    print("⚠️ Retraining model with 3-class data...")
    
    # Load datasets
    df_2class = pd.read_csv('data.csv')
    df_3class = pd.read_csv(data_3class_path)
    
    print(f"✅ 2-class dataset: {len(df_2class)} rows")
    print(f"✅ 3-class dataset: {len(df_3class)} rows")
    
    # Combine datasets
    df_combined = pd.concat([df_2class, df_3class], ignore_index=True)
    
    # Standardize column names
    if 'review' in df_combined.columns:
        df_combined.rename(columns={'review': 'text'}, inplace=True)
    if 'label' in df_combined.columns:
        df_combined.rename(columns={'label': 'sentiment'}, inplace=True)
    
    # Clean text
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)
    
    df_combined['clean_text'] = df_combined['text'].apply(clean_text)
    
    # Map sentiments to numbers (3 classes)
    df_combined['sentiment_num'] = df_combined['sentiment'].map({
        'negative': 0,
        'neutral': 1,
        'positive': 2
    })
    
    df_combined = df_combined.dropna(subset=['sentiment_num'])
    
    print(f"✅ Combined dataset: {len(df_combined)} rows")
    print("Class distribution:")
    print(df_combined['sentiment_num'].value_counts().sort_index())
    
    # Train model
    X = df_combined['clean_text']
    y = df_combined['sentiment_num']
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, model_path)
    print("✅ Model trained and saved as 'sentiment_model.pkl'")

print("Starting server...")

# Check if we need to retrain
if os.path.exists(data_3class_path):
    df_check = pd.read_csv(data_3class_path)
    if len(df_check) > min_data_rows:
        print(f"⚠️ New data detected ({len(df_check)} rows). Retraining model...")
        train_model()
    else:
        print("✅ Using cached 3-class data (small dataset). Loading existing model...")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            print("⚠️ Model not found! Training on default data...")
            train_model()
else:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("✅ Model loaded from cache.")
    else:
        print("⚠️ No model found! Training on default data...")
        train_model()

# Define Input/Output structure
class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    scores: dict

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

# Create the Prediction Endpoint
@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: TextRequest):
    try:
        clean_input = clean_text(request.text)
        
        if not clean_input:
            raise HTTPException(status_code=400, detail="Text is empty after preprocessing")

        prediction = model.predict([clean_input])[0]
        probabilities = model.predict_proba([clean_input])[0]

        # ✅ 3-Class Mapping
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        final_label = label_map[prediction]

        # ✅ Include all 3 scores
        scores = {
            "Negative": float(probabilities[0]),
            "Neutral": float(probabilities[1]),
            "Positive": float(probabilities[2])
        }

        return {
            "label": final_label,
            "scores": scores
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

print("✅ Server running on port 8000")