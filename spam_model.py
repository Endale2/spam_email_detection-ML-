import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Load the Enron dataset (replace 'enron_spam_data.csv' with your actual file path)
df = pd.read_csv('enron_spam_data.csv')

# Preprocessing: map 'Spam/Ham' to 1 for spam and 0 for ham
df['Spam/Ham'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})

# Handle NaN values: Remove rows with NaN in the 'Message' column
df = df.dropna(subset=['Message'])

# Alternatively, you could fill NaN with an empty string:
# df['Message'] = df['Message'].fillna('')

# Split data into features (X) and target (y)
X = df['Message']
y = df['Spam/Ham']

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with TF-IDF vectorizer and Logistic Regression (with Naive Bayes as an alternative)
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.9, min_df=3)),
    ('classifier', LogisticRegression(max_iter=1000))  # Change to MultinomialNB() for comparison
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('templates/confusion_matrix.png')
plt.close()

# Save the trained model
joblib.dump(model, "spam_classifier.pkl")
print("Model saved as spam_classifier.pkl")
