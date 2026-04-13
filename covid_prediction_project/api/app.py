from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Model and Scaler Paths
MODEL_PATH = os.path.join('..', 'models', 'covid_model.pkl')
SCALER_PATH = os.path.join('..', 'models', 'scaler.pkl')

# Load the model and scaler at startup
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint to determine if a patient is Covid Positive."""
    if model is None or scaler is None:
        return jsonify({'error': 'Model or Scaler not loaded on server.'}), 500

    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Required features
        features = [
            'age', 'fever', 'cough', 'breathing_difficulty', 
            'oxygen_level', 'travel_history', 'contact_with_patient'
        ]
        
        # Validate input JSON
        input_data = []
        for feature in features:
            if feature not in data:
                return jsonify({'error': f"Missing feature: {feature}"}), 400
            input_data.append(data[feature])
        
        # Convert to numpy array and reshape for single prediction
        input_array = np.array(input_data).reshape(1, -1)
        
        # Preprocess the input using the same scaler
        scaled_input = scaler.transform(input_array)
        
        # Perform prediction
        prediction = int(model.predict(scaled_input)[0])
        
        # Response logic
        message = "Covid Positive" if prediction == 1 else "Covid Negative"
        
        return jsonify({
            'prediction': prediction,
            'message': message
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Running from the 'api' folder: use cd covid_prediction_project/api; python app.py
    app.run(debug=True, port=5000)
