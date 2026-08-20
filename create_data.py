import pandas as pd

# Create sample 3-class data
data = {
    'text': [
        'This movie is amazing!',
        'It was okay, nothing special',
        'Terrible film, waste of time',
        'Pretty good movie',
        'Average, could be better',
        'Worst movie ever'
    ],
    'sentiment': [
        'positive',
        'neutral',
        'negative',
        'positive',
        'neutral',
        'negative'
    ]
}

df = pd.DataFrame(data)
df.to_csv('data_3class.csv', index=False)
print("✅ data_3class.csv created!")