import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

# Load and prepare data
df = pd.read_csv("hf://datasets/giseldo/deep-se/deep-se.csv")
df = df[df['project'] == 'appceleratorstudio']
columns = ['project', 'issuekey', 'title', 'description', 'storypoint']
df = df[columns]

# Combine title and description for text features
df['context'] = df['title'] + ' ' + df['description'].fillna('')

# Split data
X = df['context']
y = df['storypoint']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Dummy Regressor (baseline)
dummy = DummyRegressor(strategy='mean')
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
dummy_mae = mean_absolute_error(y_test, dummy_pred)

# 2. TF-IDF + Linear Regression
tfidf_lr = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('lr', LinearRegression())
])

tfidf_lr.fit(X_train, y_train)
tfidf_pred = tfidf_lr.predict(X_test)
tfidf_mae = mean_absolute_error(y_test, tfidf_pred)

# Print results
print("\nModel Comparison (MAE):")
print(f"Dummy Regressor: {dummy_mae:.2f}")
print(f"TF-IDF + Linear Regression: {tfidf_mae:.2f}")

# Calculate improvement over baseline
improvement = ((dummy_mae - tfidf_mae) / dummy_mae) * 100
print(f"\nImprovement over baseline: {improvement:.2f}%")