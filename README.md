# Sentiment Analysis Tool

## Project Overview
An AI-powered sentiment analysis tool that classifies movie reviews 
as Positive or Negative using Machine Learning.

## Dataset
- **Source:** Kaggle IMDB Movie Reviews
- **Size:** 50,000 reviews
- **Classes:** Positive (25,000), Negative (25,000)
- **Balance:** Perfectly balanced

## Preprocessing Pipeline
1. **Lowercasing:** Convert all text to lowercase
2. **HTML Removal:** Remove HTML tags like `<br />`
3. **Special Character Removal:** Remove numbers, punctuation
4. **Tokenization:** Split text into individual words
5. **Stop-word Removal:** Remove common words (the, is, a)
6. **Lemmatization:** Convert words to root form (running → run)

## Model Architecture
- **Algorithm:** Logistic Regression
- **Vectorization:** TF-IDF (5000 features)
- **Training Data:** 40,000 reviews (80%)
- **Test Data:** 10,000 reviews (20%)
- **Accuracy:** 89%

## Evaluation Metrics
- Accuracy: 89%
- Precision: ~88%
- Recall: ~89%
- F1-Score: ~0.88

## Tech Stack
- **Backend:** FastAPI, Uvicorn, Python
- **Frontend:** HTML, CSS, JavaScript
- **ML Libraries:** Scikit-learn, NLTK, Pandas
- **Deployment:** Render (Backend), Vercel (Frontend)
- **Version Control:** Git, GitHub

## Project Structure
Sentiment-backend/ ├── data.csv # 50,000 reviews ├── train_model.py # Model training script ├── sentiment_model.pkl # Trained model ├── backend_api.py # FastAPI server ├── frontend/ │ └── index.html # User interface ├── requirements.txt # Python dependencies └── README.md # This file


## How to Run Locally

### Backend
```bash
pip install -r requirements.txt
python train_model.py
uvicorn backend_api:app --reload
Frontend
Open frontend/index.html in browser

Live Demo
Frontend: https://sentiment-backend-nine.vercel.app
Backend: https://sentiment-review-api-xxcz.onrender.com
Key Learnings
Full ML pipeline implementation
NLP preprocessing techniques
REST API development
Frontend-backend integration
Cloud deployment