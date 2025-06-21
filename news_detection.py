# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load dataset
# Download the dataset from Kaggle first: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
df = pd.read_csv('Fake.csv')  # Assuming you're using the 'Fake.csv' part of dataset
df['label'] = 1  # 1 = fake

df_true = pd.read_csv('True.csv')
df_true['label'] = 0  # 0 = real

# Combine both into one dataset
data = pd.concat([df, df_true])
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle

# Step 3: Prepare data
X = data['text']
y = data['label']

# Step 4: Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Text vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Predict on your own sentence
def predict_news(news):
    news_tfidf = vectorizer.transform([news])
    pred = model.predict(news_tfidf)
    return "Fake" if pred[0] == 1 else "Real"

# Test
print("\nExample Prediction:", predict_news("Breaking news! Scientists discovered water on Mars."))