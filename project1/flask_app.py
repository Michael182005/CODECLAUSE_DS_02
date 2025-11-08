from flask import Flask, request, jsonify, send_file
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

model = joblib.load('models/model.joblib')
vectorizer = joblib.load('models/vectorizer.joblib')

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

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data['review']
    processed = preprocess_text(review)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    prob = model.predict_proba(vectorized)[0]
    prob_positive = prob[1] if prediction == 'positive' else prob[0]
    return jsonify({'label': prediction, 'prob': float(prob_positive)})

if __name__ == '__main__':
    app.run(debug=True)
