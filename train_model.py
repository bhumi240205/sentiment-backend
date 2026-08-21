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

print("Loading data...")
df = pd.read_csv('data.csv')

# Fix column names if needed
if 'review' in df.columns: df.rename(columns={'review': 'text'}, inplace=True)
if 'label' in df.columns: df.rename(columns={'label': 'sentiment'}, inplace=True)

# Clean Text Function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text) # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text) # Remove special chars
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# Map labels to numbers
df['sentiment_num'] = df['sentiment'].map({'negative': 0, 'positive': 1})
df = df.dropna(subset=['sentiment_num'])

# Split Data
X = df['clean_text']
y = df['sentiment_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])
model.fit(X_train, y_train)

# Check Accuracy
acc = model.score(X_test, y_test)
print(f"✅ Accuracy: {acc:.2f}")

# Save Model
joblib.dump(model, 'sentiment_model.pkl')
print("✅ Model saved as 'sentiment_model.pkl'")