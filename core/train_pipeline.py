import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from .config import (
    MODEL_DIR, TEST_SIZE, RANDOM_STATE,
    TFIDF_MAX_FEATURES, LOGREG_MAX_ITER
)
from .data_loader import load_balanced_subset


def build_and_save_model():
    """
    Non-interactive end-to-end pipeline:
    load data -> split -> vectorize -> train -> evaluate -> save artifacts
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading balanced subset...")
    X, y = load_balanced_subset()

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=LOGREG_MAX_ITER)
    model.fit(X_train_vec, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred)
    print(report)

    print("Saving artifacts...")
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(model, os.path.join(MODEL_DIR, "sentiment_model.joblib"))

    print("Done. Artifacts created under model_artifacts/")
    return report
