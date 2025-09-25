# titanic_logistic_demo.py
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore
import matplotlib.pyplot as plt
import numpy as np

# Load dataset (built-in)
df = sns.load_dataset('titanic')

# Quick inspection
print(df.head())
print(df.info())

# Select features and target
cols = ['pclass','sex','age','sibsp','parch','fare','embarked']
df = df[cols + ['survived']].copy()

# Handle missing values
df['age'] = df['age'].fillna(df['age'].median())
df = df.dropna(subset=['embarked'])  # small number of rows

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['sex','embarked'], drop_first=True)

# Features and target
X = df.drop('survived', axis=1)
y = df['survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix - Titanic')
plt.tight_layout()
plt.savefig('titanic_confusion_matrix.png', dpi=200)
plt.show()
