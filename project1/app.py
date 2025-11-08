import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Load and train model
@st.cache_data
def load_and_train_model():
    df = pd.read_csv('movie_reviews.csv')
    df['processed_review'] = df['review'].apply(preprocess_text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_review'])
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, vectorizer, X_test, y_test

# Predict sentiment
def predict_sentiment(review, model, vectorizer):
    processed = preprocess_text(review)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    return prediction

# Streamlit app
st.title("Movie Review Sentiment Analysis")

# Load model
model, vectorizer, X_test, y_test = load_and_train_model()

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.header("Model Evaluation")
st.write(f"Accuracy: {accuracy:.2f}")
st.write("Confusion Matrix:")
st.write(cm)

# User input
st.header("Predict Sentiment")
user_review = st.text_area("Enter a movie review:")
if st.button("Predict"):
    if user_review:
        prediction = predict_sentiment(user_review, model, vectorizer)
        st.write(f"Predicted Sentiment: {prediction}")
    else:
        st.write("Please enter a review.")
