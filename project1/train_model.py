import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

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

def build_pipeline(df):
    df['processed_review'] = df['review'].apply(preprocess_text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_review'])
    y = df['sentiment']
    return X, y, vectorizer

def train_eval(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, report, cm

def save_artifacts(model, vectorizer, folder):
    os.makedirs(folder, exist_ok=True)
    joblib.dump(model, os.path.join(folder, 'model.joblib'))
    joblib.dump(vectorizer, os.path.join(folder, 'vectorizer.joblib'))

def main():
    df = load_data('movie_reviews.csv')
    X, y, vectorizer = build_pipeline(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'LogisticRegression': LogisticRegression(),
        'MultinomialNB': MultinomialNB()
    }

    best_acc = 0
    best_model = None
    best_name = None

    for name, model in models.items():
        acc, report, cm = train_eval(X_train, y_train, X_test, y_test, model)
        print(f"{name} Accuracy: {acc:.2f}")
        print(f"Classification Report:\n{report}")
        print(f"Confusion Matrix:\n{cm}\n")
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    print(f"Best model: {best_name} with accuracy {best_acc:.2f}")
    save_artifacts(best_model, vectorizer, 'models')

if __name__ == "__main__":
    main()
