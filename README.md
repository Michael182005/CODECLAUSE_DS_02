🎥 Movie Genre Prediction

Internship Project — CodeClause | Data Science Domain

🧠 Overview

This project predicts the genre of a movie based on its description or synopsis using Natural Language Processing (NLP) and Machine Learning techniques.
The model learns from a dataset of movie plots and corresponding genres, allowing users to input new movie descriptions and instantly see predicted genres through a Streamlit-based UI.

🚀 Features

Interactive Streamlit web app for predictions

Text preprocessing pipeline using NLP (tokenization, stopword removal, lemmatization)

TF-IDF vectorization for feature extraction

Multi-class genre classification model (e.g., Action, Comedy, Drama, Thriller, etc.)

Option to input single movie description or upload CSV for batch prediction

Displays prediction probabilities and confidence for each input

🧩 Project Structure
📁 Movie-Genre-Prediction
│
├── 📄 train_model.py           # Model training and saving script
├── 📄 app.py                   # Streamlit app for prediction
├── 📁 models/                  # Trained model and TF-IDF vectorizer
│   ├── model.joblib
│   └── vectorizer.joblib
├── 📄 dataset.csv              # Dataset with movie description and genre
├── 📄 requirements.txt         # Required Python libraries
└── 📄 README.md                # Project documentation

⚙️ Technologies Used

Python 3.10+

Pandas & NumPy – Data manipulation

Scikit-learn – ML algorithms and model evaluation

NLTK / SpaCy – Text preprocessing

Streamlit – Interactive user interface

Joblib – Model serialization

🧰 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/movie-genre-prediction.git
cd movie-genre-prediction

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate    # for macOS/Linux
venv\Scripts\activate       # for Windows

3️⃣ Install Required Packages
pip install -r requirements.txt

🧪 Model Training

Run this command to train and save your genre prediction model:

python train_model.py


The training script will:

Clean and preprocess text data (remove stopwords, punctuation, lowercase, lemmatize).

Convert text data into numeric vectors using TF-IDF.

Train a classification model (Logistic Regression or Random Forest).

Evaluate model accuracy and save the model + vectorizer in /models.

🌐 Run the Streamlit App

Once the model is trained, start the web app using:

streamlit run app.py

🖥️ App Usage

🔹 Single Description Mode:
Enter a short movie description in the text box and click Predict Genre.
The app will display the predicted genre and its confidence level.

🔹 Batch Mode:
Upload a CSV file with a column named description.
The app will predict genres for all entries and display a table with prediction results and probabilities.

📊 Example Output
Movie Description	Predicted Genre	Confidence
“A retired hitman returns to seek revenge.”	Action	0.92
“Two best friends discover their true feelings at summer camp.”	Romance	0.87
“A detective investigates a mysterious murder in a small town.”	Thriller	0.90
📚 Learning Outcomes

Text preprocessing & NLP for classification tasks

Feature extraction using TF-IDF

Building and evaluating multi-class ML models

Integrating ML models into Streamlit web apps

🧑‍💻 Author

Karthikeyan T
Data Science Intern @ CodeClause
