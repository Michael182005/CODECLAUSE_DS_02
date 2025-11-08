import streamlit as st
import pandas as pd
import joblib
import json
from utils import preprocess_text

st.set_page_config(page_title="Movie Genre Prediction — CodeClause", layout="wide")

@st.cache_resource
def load_models():
    vectorizer = joblib.load('./models/vectorizer.joblib')
    model = joblib.load('./models/model.joblib')
    with open('./models/meta.json', 'r') as f:
        meta = json.load(f)
    return vectorizer, model, meta

vectorizer, model, meta = load_models()

st.markdown("""
# Movie Genre Prediction — CodeClause

How to run:
1. `python train_genre_model.py`
2. `streamlit run app.py`
""")

mode = st.sidebar.radio("Mode", ["Single Plot", "Batch (CSV)"])

if mode == "Single Plot":
    plot = st.text_area("Enter plot summary", height=150)
    if st.button("Predict"):
        if plot:
            processed = preprocess_text(plot)
            X = vectorizer.transform([processed])
            if meta['mode'] == 'single-label':
                pred = model.predict(X)[0]
                prob = model.decision_function(X)[0] if hasattr(model, 'decision_function') else model.predict_proba(X)[0]
                st.write(f"Predicted Genre: {pred}")
                st.write(f"Score: {prob}")
            else:
                probs = model.predict_proba(X)[0]
                top_k = st.slider("Top K", 1, len(meta['labels']), 3)
                top_indices = probs.argsort()[-top_k:][::-1]
                top_genres = [meta['labels'][i] for i in top_indices]
                top_scores = [probs[i] for i in top_indices]
                st.write("Top Genres:")
                for g, s in zip(top_genres, top_scores):
                    st.write(f"{g}: {s:.4f}")
        else:
            st.error("Please enter a plot summary.")

elif mode == "Batch (CSV)":
    uploaded_file = st.file_uploader("Upload CSV with 'plot' column", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'plot' in df.columns:
            df['processed_plot'] = df['plot'].apply(preprocess_text)
            X = vectorizer.transform(df['processed_plot'])
            if meta['mode'] == 'single-label':
                preds = model.predict(X)
                df['predicted'] = preds
                df['top_scores'] = [model.decision_function([x])[0] if hasattr(model, 'decision_function') else model.predict_proba([x])[0] for x in X]
            else:
                probs = model.predict_proba(X)
                preds = []
                top_scores = []
                for prob in probs:
                    top_indices = prob.argsort()[-3:][::-1]
                    preds.append(', '.join([meta['labels'][i] for i in top_indices]))
                    top_scores.append({meta['labels'][i]: prob[i] for i in top_indices})
                df['predicted'] = preds
                df['top_scores'] = top_scores
            st.dataframe(df[['plot', 'predicted', 'top_scores']])
            genre_counts = df['predicted'].value_counts()
            st.bar_chart(genre_counts)
        else:
            st.error("CSV must have a 'plot' column.")
