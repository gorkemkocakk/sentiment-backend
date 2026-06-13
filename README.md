# Sentiment Analysis Backend

This repository contains the backend API for a sentiment analysis project. The application takes a text input and predicts whether the sentiment is positive or negative using a machine learning model.

## Project Overview

The goal of this project is to build a simple end-to-end sentiment analysis system with a backend API that can serve machine learning predictions.

The backend was developed with FastAPI and uses a trained machine learning model based on TF-IDF vectorization and Logistic Regression.

## Features

* REST API built with FastAPI
* Sentiment prediction endpoint
* Text preprocessing and vectorization
* Machine learning model using TF-IDF and Logistic Regression
* Docker support for deployment
* Simple request/response structure for integration with a frontend application

## Tech Stack

* Python
* FastAPI
* Scikit-learn
* TF-IDF
* Logistic Regression
* Docker

## API Endpoint

### Predict Sentiment

```http
POST /predict
```

Example request:

```json
{
  "text": "This movie was really good and enjoyable."
}
```

Example response:

```json
{
  "sentiment": "positive"
}
```

## How to Run Locally

Clone the repository:

```bash
git clone https://github.com/gorkemkocakk/sentiment-backend.git
cd sentiment-backend
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## Project Status

This project is an educational machine learning backend project developed to practice API development, machine learning model serving, and deployment preparation.

## Author

Oğuzhan Görkem Koçak
Management Information Systems Graduate
GitHub: gorkemkocakk
