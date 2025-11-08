import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import json
import os
from utils import preprocess_text

def load_data(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'data', 'movies.csv')
    df = pd.read_csv(path)
    return df[['plot', 'genre']]

def detect_label_mode(df):
    genres = df['genre'].str.contains('|')
    if genres.any():
        return 'multi-label'
    return 'single-label'

def build_pipeline(mode, vectorizer):
    if mode == 'single-label':
        models = {
            'LinearSVC': LinearSVC(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        return models
    else:
        return OneVsRestClassifier(LogisticRegression(random_state=42, max_iter=1000))

def train_eval(df, mode, vectorizer, pipeline):
    df['processed_plot'] = df['plot'].apply(preprocess_text)
    X = vectorizer.fit_transform(df['processed_plot'])
    if mode == 'single-label':
        y = df['genre']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        best_model = None
        best_f1 = 0
        for name, model in pipeline.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
        y_pred = best_model.predict(X_test)
        print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
        print(f'Macro F1: {f1_score(y_test, y_pred, average="macro"):.4f}')
        print(classification_report(y_test, y_pred))
        return best_model, vectorizer
    else:
        y = df['genre'].str.get_dummies(sep='|')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
        print(f'Macro F1: {f1_score(y_test, y_pred, average="macro"):.4f}')
        print(classification_report(y_test, y_pred, target_names=y.columns))
        return pipeline, vectorizer

def save_artifacts(model, vectorizer, mode, labels, path='./models'):
    os.makedirs(path, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(path, 'vectorizer.joblib'))
    joblib.dump(model, os.path.join(path, 'model.joblib'))
    meta = {'mode': mode, 'labels': labels}
    with open(os.path.join(path, 'meta.json'), 'w') as f:
        json.dump(meta, f)

def main():
    df = load_data()
    mode = detect_label_mode(df)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=5000)
    pipeline = build_pipeline(mode, vectorizer)
    model, vectorizer = train_eval(df, mode, vectorizer, pipeline)
    labels = df['genre'].unique().tolist() if mode == 'single-label' else df['genre'].str.split('|').explode().unique().tolist()
    save_artifacts(model, vectorizer, mode, labels)

if __name__ == '__main__':
    main()
