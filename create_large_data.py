import pandas as pd
import random

# Create a balanced 3-class dataset
positive_samples = [
    "This movie is amazing!", "Absolutely fantastic film!", "Best movie ever made!",
    "Outstanding performance!", "Highly recommended!", "A masterpiece!",
    "Loved every minute of it!", "Brilliant storytelling!", "Great cinematography!",
    "Perfect plot and acting!"
]

negative_samples = [
    "Terrible waste of time!", "Worst movie I have ever seen!", "Horrible acting and plot!",
    "Boring and dull!", "A complete disaster!", "Do not watch this!",
    "Awful experience!", "Poor quality film!", "Disappointing ending!",
    "Total waste of money!"
]

neutral_samples = [
    "The movie was okay.", "It was just average.", "Nothing special about this film.",
    "Decent but not great.", "Neither good nor bad.", "Average experience.",
    "Could be better.", "Not bad, not good.", "Meh, it was fine.", "Just okay.",
    "Passable film.", "Nothing to write home about.", "Standard movie.",
    "Average at best.", "Could have been better.", "Not memorable.",
    "Ordinary film.", "Nothing extraordinary.", "Mediocre at best.", "Just fine."
]

# Create dataset with balanced classes
data = {
    'text': [],
    'sentiment': []
}

# Add 500 positive samples
for _ in range(500):
    data['text'].append(random.choice(positive_samples))
    data['sentiment'].append('positive')

# Add 500 negative samples
for _ in range(500):
    data['text'].append(random.choice(negative_samples))
    data['sentiment'].append('negative')

# Add 500 neutral samples (CRITICAL!)
for _ in range(500):
    data['text'].append(random.choice(neutral_samples))
    data['sentiment'].append('neutral')

df = pd.DataFrame(data)
df.to_csv('data_3class.csv', index=False)

print(f"✅ New dataset created with {len(df)} rows!")
print("Class distribution:")
print(df['sentiment'].value_counts())