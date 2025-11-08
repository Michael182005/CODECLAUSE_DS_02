"""
README:
Run 'python train_model.py' once to train and save the model.
Then run 'streamlit run app_new.py' to start the UI.
"""

import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import plotly.express as px

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

@st.cache_resource
def load_model():
    model = joblib.load('models/model.joblib')
    vectorizer = joblib.load('models/vectorizer.joblib')
    return model, vectorizer

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

def predict_sentiment(review, model, vectorizer):
    processed = preprocess_text(review)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    prob = model.predict_proba(vectorized)[0]
    prob_positive = prob[1] if prediction == 'positive' else prob[0]
    return prediction, prob_positive

st.title("Movie Review Sentiment – CodeClause")

model, vectorizer = load_model()

mode = st.sidebar.radio("Mode", ["Single Review", "Batch (CSV)"])

if mode == "Single Review":
    st.header("Analyze a Single Review")
    review = st.text_area("Paste your movie review here:")
    if st.button("Analyze"):
        if review:
            prediction, prob = predict_sentiment(review, model, vectorizer)
            emoji = "😊" if prediction == "positive" else "😞"
            color = "green" if prediction == "positive" else "red"
            st.markdown(f"<div style='background-color:{color}; padding:10px; border-radius:5px;'>Predicted: {prediction} {emoji}<br>Probability: {prob:.2f}</div>", unsafe_allow_html=True)
        else:
            st.write("Please enter a review.")

elif mode == "Batch (CSV)":
    st.header("Analyze Batch Reviews from CSV")
    uploaded_file = st.file_uploader("Upload CSV with 'review' column", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'review' in df.columns:
            results = []
            for review in df['review']:
                pred, prob = predict_sentiment(review, model, vectorizer)
                results.append({'review': review, 'prediction': pred, 'probability': prob})
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            counts = results_df['prediction'].value_counts()
            fig = px.bar(counts, x=counts.index, y=counts.values, title="Sentiment Counts")
            st.plotly_chart(fig)
        else:
            st.write("CSV must have a 'review' column.")
