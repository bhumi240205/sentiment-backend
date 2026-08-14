from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <--- Yeh line add karein
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize FastAPI app
app = FastAPI(title="Sentiment Analysis API")

# --- CORS Configuration (New) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS)
    allow_headers=["*"],  # Allows all headers
)
# -------------------------------

# Load NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('lemmatizers/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load the trained model
print("Loading model...")
try:
    model = joblib.load('sentiment_model.pkl')
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'sentiment_model.pkl' file nahi mili! Step 1 complete karein.")
    exit()

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

