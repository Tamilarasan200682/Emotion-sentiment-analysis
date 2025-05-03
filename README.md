# Phase 2: Model Evaluation, Visualization, Deployment for Sentiment Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import tensorflow as tf

# Load test data
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').values.ravel()

# Load trained models
lr_model = joblib.load('models/lr_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
lstm_model = tf.keras.models.load_model('models/sentiment_lstm.h5')

# Evaluate Logistic Regression
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))

# Plot Confusion Matrix for LR
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/lr_confusion_matrix.png")
plt.close()

# Plot Confusion Matrix for RF
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Greens")
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/rf_confusion_matrix.png")
plt.close()

# Evaluate LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = joblib.load('models/tokenizer.pkl')
X_test_text = pd.read_csv('data/X_test_text.csv')
X_test_seq = tokenizer.texts_to_sequences(X_test_text['text'])
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

lstm_pred = lstm_model.predict(X_test_pad)
lstm_pred_classes = lstm_pred.argmax(axis=1)
print("LSTM Evaluation:")
print(classification_report(y_test, lstm_pred_classes))

# Save evaluation reports
with open("outputs/evaluation_report.txt", "w") as f:
    f.write("Logistic Regression Report:\n")
    f.write(classification_report(y_test, y_pred_lr))
    f.write("\nRandom Forest Report:\n")
    f.write(classification_report(y_test, y_pred_rf))
    f.write("\nLSTM Report:\n")
    f.write(classification_report(y_test, lstm_pred_classes))# Phase 2: Model Evaluation, Visualization, Deployment for Sentiment Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import tensorflow as tf

# Load test data
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').values.ravel()

# Load trained models
lr_model = joblib.load('models/lr_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
lstm_model = tf.keras.models.load_model('models/sentiment_lstm.h5')

# Evaluate Logistic Regression
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))

# Plot Confusion Matrix for LR
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/lr_confusion_matrix.png")
plt.close()

# Plot Confusion Matrix for RF
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Greens")
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/rf_confusion_matrix.png")
plt.close()

# Evaluate LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = joblib.load('models/tokenizer.pkl')
X_test_text = pd.read_csv('data/X_test_text.csv')
X_test_seq = tokenizer.texts_to_sequences(X_test_text['text'])
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

lstm_pred = lstm_model.predict(X_test_pad)
lstm_pred_classes = lstm_pred.argmax(axis=1)
print("LSTM Evaluation:")
print(classification_report(y_test, lstm_pred_classes))

# Save evaluation reports
with open("outputs/evaluation_report.txt", "w") as f:
    f.write("Logistic Regression Report:\n")
    f.write(classification_report(y_test, y_pred_lr))
    f.write("\nRandom Forest Report:\n")
    f.write(classification_report(y_test, y_pred_rf))
    f.write("\nLSTM Report:\n")
    f.write(classification_report(y_test, lstm_pred_classes))# Phase 2: Model Evaluation, Visualization, Deployment for Sentiment Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import tensorflow as tf

# Load test data
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').values.ravel()

# Load trained models
lr_model = joblib.load('models/lr_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
lstm_model = tf.keras.models.load_model('models/sentiment_lstm.h5')

# Evaluate Logistic Regression
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))

# Plot Confusion Matrix for LR
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/lr_confusion_matrix.png")
plt.close()

# Plot Confusion Matrix for RF
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Greens")
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/rf_confusion_matrix.png")
plt.close()

# Evaluate LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = joblib.load('models/tokenizer.pkl')
X_test_text = pd.read_csv('data/X_test_text.csv')
X_test_seq = tokenizer.texts_to_sequences(X_test_text['text'])
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

lstm_pred = lstm_model.predict(X_test_pad)
lstm_pred_classes = lstm_pred.argmax(axis=1)
print("LSTM Evaluation:")
print(classification_report(y_test, lstm_pred_classes))

# Save evaluation reports
with open("outputs/evaluation_report.txt", "w") as f:
    f.write("Logistic Regression Report:\n")
    f.write(classification_report(y_test, y_pred_lr))
    f.write("\nRandom Forest Report:\n")
    f.write(classification_report(y_test, y_pred_rf))
    f.write("\nLSTM Report:\n")
    f.write(classification_report(y_test, lstm_pred_classes))
