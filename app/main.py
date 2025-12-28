from fastapi import FastAPI
import joblib
import os

from app.schemas import SentimentRequest, SentimentResponse

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment of text using a trained ML model",
    version="1.0.0"
)

# Load model artifacts
MODEL_DIR = "model_artifacts"
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
model = joblib.load(os.path.join(MODEL_DIR, "sentiment_model.joblib"))


@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment for a given input text.

    - **text**: input review text
    - returns predicted sentiment label
    """
    X_vec = vectorizer.transform([request.text])
    prediction = model.predict(X_vec)[0]

    return SentimentResponse(sentiment=prediction)
