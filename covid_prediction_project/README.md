# COVID-19 Prediction Project

This project implements a complete machine learning pipeline using Logistic Regression to predict COVID-19 cases from synthetic symptoms data.

## Features:
- Synthetic Data Generation: Generates a CSV dataset with 500 rows.
- Data Loading: Loads data from a local CSV file.
- Data Preprocessing: Handles train-test split and feature scaling.
- Logistic Regression Training: Trains the model on symptom features.
- Model Evaluation: Reports accuracy, confusion matrix, and classification results.
- Model Persistence: Saves the trained model and scaler to disk.

## Project Structure:
- `data/`: Contains the generated `covid.csv`.
- `models/`: Contains the saved `logistic_model.pkl` and `scaler.pkl`.
- `src/`: Modular code for loading, preprocessing, training, predicting, and utilities.
- `main.py`: Main script to run the complete pipeline.
- `requirements.txt`: Project dependencies.

## API (Flask):
- `api/app.py`: Provides a POST `/predict` endpoint.
- Loads `covid_model.pkl` and `scaler.pkl` at startup.

## Installation:
```bash
pip install -r requirements.txt
```

## Running the Pipeline (Retrain):
```bash
python main.py
```

## Running the API:
```bash
cd api
python app.py
```

## Testing the API:
Aap is `curl` command se check kar sakte hain:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"age\": 40, \"fever\": 1, \"cough\": 1, \"breathing_difficulty\": 0, \"oxygen_level\": 96, \"travel_history\": 0, \"contact_with_patient\": 1}"
```
