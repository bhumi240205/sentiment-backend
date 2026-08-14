# Step 1: Data load aur explore karna
import pandas as pd

# Data load karo
df = pd.read_csv('data.csv')

# Data ki basic info dekho
print("Total rows:", len(df))
print("\nColumn names:", df.columns.tolist())
print("\nPehle 5 rows:")
print(df.head())

# Sentiment distribution dekho (kitne positive, kitne negative)
print("\nSentiment distribution:")
print(df['sentiment'].value_counts())