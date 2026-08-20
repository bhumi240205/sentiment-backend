import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download NLTK data (first time run only)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('lemmatizers/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


print("Loading datasets...")

# Load BOTH datasets
df_2class = pd.read_csv('data.csv')
df_3class = pd.read_csv('data_3class.csv')

print(f"2-class dataset size: {len(df_2class)}")
print(f"3-class dataset size: {len(df_3class)}")

# Display unique sentiments in each dataset
print("\n2-class sentiments:", df_2class['sentiment'].unique() if 'sentiment' in df_2class.columns else df_2class['label'].unique())
print("3-class sentiments:", df_3class['sentiment'].unique() if 'sentiment' in df_3class.columns else df_3class['label'].unique())

# Choose which dataset to use for training
USE_3CLASS = True  # Set to False to use 2-class dataset

if USE_3CLASS:
    print("\n📊 Training with 3-CLASS dataset (Positive, Negative, Neutral)")
    df = df_3class
else:
    print("\n📊 Training with 2-CLASS dataset (Positive, Negative)")
    df = df_2class


# Fix column names if needed
if 'review' in df.columns:
    df.rename(columns={'review': 'text'}, inplace=True)
if 'label' in df.columns:
    df.rename(columns={'label': 'sentiment'}, inplace=True)


# Clean Text Function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


print("\nCleaning text...")
df['clean_text'] = df['text'].apply(clean_text)


# Map labels to numbers based on dataset type
if USE_3CLASS:
    df['sentiment_num'] = df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
else:
    df['sentiment_num'] = df['sentiment'].map({'negative': 0, 'positive': 1})

df = df.dropna(subset=['sentiment_num'])

# Show class distribution
print("\nClass distribution:")
print(df['sentiment_num'].value_counts().sort_index())


# Split Data
X = df['clean_text']
y = df['sentiment_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Train Model
print("\nTraining model...")
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])
model.fit(X_train, y_train)


# Check Accuracy
acc = model.score(X_test, y_test)
print(f"\n✅ Accuracy: {acc:.2f}")


# Save Model
joblib.dump(model, 'sentiment_model.pkl')
print("✅ Model saved as 'sentiment_model.pkl'")

print("\n🎉 Training complete! Your model now supports 3 classes (Positive, Negative, Neutral)")