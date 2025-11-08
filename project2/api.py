from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
from utils import preprocess_text

app = FastAPI()

class PlotRequest(BaseModel):
    plot: str

@app.on_event("startup")
def load_models():
    global vectorizer, model, meta
    vectorizer = joblib.load('./models/vectorizer.joblib')
    model = joblib.load('./models/model.joblib')
    with open('./models/meta.json', 'r') as f:
        meta = json.load(f)

@app.post("/predict")
def predict(request: PlotRequest):
    processed = preprocess_text(request.plot)
    X = vectorizer.transform([processed])
    if meta['mode'] == 'single-label':
        pred = model.predict(X)[0]
        score = model.decision_function(X)[0] if hasattr(model, 'decision_function') else model.predict_proba(X)[0]
        return {"predicted": pred, "scores": score}
    else:
        probs = model.predict_proba(X)[0]
        scores = {meta['labels'][i]: float(probs[i]) for i in range(len(meta['labels']))}
        predicted = [meta['labels'][i] for i in probs.argsort()[-3:][::-1]]
        return {"predicted": predicted, "scores": scores}
