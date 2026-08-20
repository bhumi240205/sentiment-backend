import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('lemmatizers/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

print("Loading datasets...")

# Load datasets
df_2class = pd.read_csv('data.csv')
df_3class = pd.read_csv('data_3class.csv')

print(f"✅ 2-class dataset loaded: {len(df_2class)} rows")
print(f"✅ 3-class dataset loaded: {len(df_3class)} rows")

# Standardize column names FIRST
if 'review' in df_2class.columns:
    df_2class.rename(columns={'review': 'text'}, inplace=True)
if 'label' in df_2class.columns:
    df_2class.rename(columns={'label': 'sentiment'}, inplace=True)

if 'review' in df_3class.columns:
    df_3class.rename(columns={'review': 'text'}, inplace=True)
if 'label' in df_3class.columns:
    df_3class.rename(columns={'label': 'sentiment'}, inplace=True)

# Create a NEW DataFrame with clean index from scratch
print("\nBuilding combined dataset...")
combined_text = []
combined_sentiment = []

# Add 2-class data
combined_text.extend(df_2class['text'].tolist())
combined_sentiment.extend(df_2class['sentiment'].tolist())

# Add 3-class data
combined_text.extend(df_3class['text'].tolist())
combined_sentiment.extend(df_3class['sentiment'].tolist())

# Create fresh DataFrame with automatic clean index
df_combined = pd.DataFrame({
    'text': combined_text,
    'sentiment': combined_sentiment
})

print(f"✅ Combined dataset size: {len(df_combined)} rows")
print("Unique sentiments:", df_combined['sentiment'].unique())

# Clean text function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

print("\nCleaning text...")
df_combined['clean_text'] = df_combined['text'].apply(clean_text)

# Map sentiments
df_combined['sentiment_num'] = df_combined['sentiment'].map({
    'negative': 0,
    'neutral': 1,
    'positive': 2
})

df_combined = df_combined.dropna(subset=['sentiment_num'])

print("\nClass distribution:")
print(df_combined['sentiment_num'].value_counts().sort_index())

# Split data
X = df_combined['clean_text']
y = df_combined['sentiment_num']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
print("\nTraining 3-class model...")
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"✅ Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, 'sentiment_model.pkl')
print("✅ Model saved as 'sentiment_model.pkl'")

print("\n🎉 Training complete! Model now supports 3 classes (Negative, Neutral, Positive)")