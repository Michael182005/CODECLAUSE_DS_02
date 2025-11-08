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

# Download NLTK resources if not present
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, vectorizer):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm, y_pred

def main():
    # Load dataset
    df = pd.read_csv('movie_reviews.csv')
    df['processed_review'] = df['review'].apply(preprocess_text)

    # Vectorize
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_review'])
    y = df['sentiment']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    accuracy, cm, y_pred = evaluate_model(model, X_test, y_test, vectorizer)

    # Print summary
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nExample Predictions:")
    for i in range(min(5, len(y_test))):
        print(f"Review: {df.iloc[y_test.index[i]]['review'][:50]}...")
        print(f"Actual: {y_test.iloc[i]}, Predicted: {y_pred[i]}")
        print()

if __name__ == "__main__":
    main()
