import joblib
import pandas as pd

def load_saved_model(model_path):
    """Loads a pre-trained model from disk."""
    model = joblib.load(model_path)
    print(f"Model loaded: {model_path}")
    return model

def predict_on_new_data(model, X_new_data):
    """Predicts outcome for new data."""
    prediction = model.predict(X_new_data)
    print(f"Prediction result: {prediction}")
    return prediction
