import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor
from datasets import load_dataset
from tabulate import tabulate
import textstat
import re
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data
nltk.download('punkt')

# Load the dataset
ds = load_dataset("giseldo/neodataset")
df = ds["train"].to_pandas()

# Remove rows with NA values in storypoints, title or description
df = df.dropna(subset=['storypoints', 'title', 'description'])
print(f"Number of rows after removing NA values: {len(df)}")

# Function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):
        return text.lower()
    return ""

# Function to extract readability features
def extract_readability_features(text):
    if not isinstance(text, str):
        return {
            'gunning_fog': 0,
            'text_length': 0,
            'sentence_count': 0,
            'word_count': 0
        }
    
    return {
        'gunning_fog': textstat.gunning_fog(text),
        'text_length': len(text),
        'sentence_count': len(re.split(r'[.!?]+', text)),
        'word_count': len(text.split())
    }

# Combine title and description
df['combined_text'] = df['title'] + ' ' + df['description']

# Prepare text for Word2Vec
tokenized_texts = [word_tokenize(preprocess_text(text)) for text in df['combined_text']]

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_texts,
                         vector_size=100,
                         window=5,
                         min_count=1,
                         workers=4)

# Function to get document vector using Word2Vec
def get_document_vector(text, model):
    words = word_tokenize(preprocess_text(text))
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    return np.zeros(model.vector_size)

# Create Word2Vec features
word2vec_features = np.array([get_document_vector(text, word2vec_model) for text in df['combined_text']])

# Extract readability features
readability_features = df['combined_text'].apply(extract_readability_features)
readability_df = pd.DataFrame(readability_features.tolist())

# Create TF-IDF features
tfidf = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    preprocessor=preprocess_text
)

# Prepare features and target
X_tfidf = tfidf.fit_transform(df['combined_text'])
X_readability = readability_df.values
X_word2vec = word2vec_features
y = df['storypoints']

# Split the data
X_tfidf_train, X_tfidf_test, X_readability_train, X_readability_test, X_word2vec_train, X_word2vec_test, y_train, y_test = train_test_split(
    X_tfidf, X_readability, X_word2vec, y, test_size=0.2, random_state=42
)

# Train the dummy model (mean predictor)
dummy_model = DummyRegressor(strategy='mean')
dummy_model.fit(X_tfidf_train, y_train)
dummy_pred = dummy_model.predict(X_tfidf_test)
dummy_mae = mean_absolute_error(y_test, dummy_pred)

# Train the TF-IDF + Linear Regression model
tfidf_model = LinearRegression()
tfidf_model.fit(X_tfidf_train, y_train)
tfidf_pred = tfidf_model.predict(X_tfidf_test)
tfidf_mae = mean_absolute_error(y_test, tfidf_pred)

# Train the Readability + Linear Regression model
readability_model = LinearRegression()
readability_model.fit(X_readability_train, y_train)
readability_pred = readability_model.predict(X_readability_test)
readability_mae = mean_absolute_error(y_test, readability_pred)

# Train the Word2Vec + Linear Regression model
word2vec_model_reg = LinearRegression()
word2vec_model_reg.fit(X_word2vec_train, y_train)
word2vec_pred = word2vec_model_reg.predict(X_word2vec_test)
word2vec_mae = mean_absolute_error(y_test, word2vec_pred)

# Create comparison table
comparison_table = [
    ["Model", "MAE"],
    ["Dummy (Mean)", f"{dummy_mae:.2f}"],
    ["TF-IDF + Linear Regression", f"{tfidf_mae:.2f}"],
    ["Readability + Linear Regression", f"{readability_mae:.2f}"],
    ["Word2Vec + Linear Regression", f"{word2vec_mae:.2f}"]
]

print("\nModel Comparison:")
print(tabulate(comparison_table, headers="firstrow", tablefmt="grid"))

# Function to predict story points using TF-IDF model
def predict_story_points_tfidf(user_story_title, user_story_description=""):
    # Combine title and description
    combined_text = user_story_title + ' ' + user_story_description
    # Transform the input text
    X_new = tfidf.transform([combined_text])
    # Make prediction
    prediction = tfidf_model.predict(X_new)
    return prediction[0]

# Function to predict story points using Readability model
def predict_story_points_readability(user_story_title, user_story_description=""):
    # Combine title and description
    combined_text = user_story_title + ' ' + user_story_description
    # Extract readability features
    features = extract_readability_features(combined_text)
    # Make prediction
    prediction = readability_model.predict([list(features.values())])
    return prediction[0]

# Function to predict story points using Word2Vec model
def predict_story_points_word2vec(user_story_title, user_story_description=""):
    # Combine title and description
    combined_text = user_story_title + ' ' + user_story_description
    # Get document vector
    doc_vector = get_document_vector(combined_text, word2vec_model)
    # Make prediction
    prediction = word2vec_model_reg.predict([doc_vector])
    return prediction[0]

# Example usage
example_title = "As a user, I want to be able to log in to the system"
example_description = "The system should allow users to log in using their email and password. After successful login, they should be redirected to their dashboard."

# Get predictions from all models
tfidf_prediction = predict_story_points_tfidf(example_title, example_description)
readability_prediction = predict_story_points_readability(example_title, example_description)
word2vec_prediction = predict_story_points_word2vec(example_title, example_description)

print(f"\nExample predictions:")
print(f"User Story Title: {example_title}")
print(f"User Story Description: {example_description}")
print(f"TF-IDF Model Prediction: {tfidf_prediction:.2f}")
print(f"Readability Model Prediction: {readability_prediction:.2f}")
print(f"Word2Vec Model Prediction: {word2vec_prediction:.2f}") 