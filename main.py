import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load your dataset (replace with actual path)
# df = pd.read_csv("data.csv")
# For placeholder:
df = pd.DataFrame({"text": ["I am happy", "I am angry", "I am sad"], "emotion": ["joy", "anger", "sadness"]})

# Preprocessing
X = df["text"]
y = df["emotion"]

# Feature Extraction
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
