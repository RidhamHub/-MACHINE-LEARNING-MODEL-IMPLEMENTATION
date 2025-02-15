import pandas as pd
import string

# Sample dataset
data = {'message': ['Win a free iPhone now!', 'Hey, how are you?', 'Congratulations! You won a lottery!', 'Are we still on for lunch?', 'Click here to claim your prize!'],
        'label': [1, 0, 1, 0, 1]}
df = pd.DataFrame(data)

# Simple text preprocessing
def clean_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text.split()

# Convert messages to feature dictionaries
X = [clean_text(msg) for msg in df['message']]
y = df['label'].tolist()

# Train a simple model (word occurrence rule-based)
def classify_message(message):
    spam_keywords = {'win', 'free', 'congratulations', 'prize', 'click', 'lottery', 'cash', 'offer'}
    words = set(clean_text(message))
    return int(bool(words & spam_keywords))

# Predict labels
df['predicted_label'] = df['message'].apply(classify_message)

# Evaluate accuracy
accuracy = sum(df['label'] == df['predicted_label']) / len(df)
print(f"Accuracy: {accuracy:.2f}")

# Save predictions to CSV
df.to_csv("spam_predictions.csv", index=False)
