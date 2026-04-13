from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train_model import train_logistic_model, evaluate_model
from src.utils import save_trained_model
import os

def run_pipeline():
    """Runs the complete ML pipeline: load, preprocess, train, evaluate, and save."""
    # File paths
    data_path = 'data/covid.csv'
    model_dir = 'models'
    
    # Load data
    df = load_data(data_path)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train model
    model = train_logistic_model(X_train, y_train)
    
    # Evaluate model
    accuracy, conf_matrix, class_report = evaluate_model(model, X_test, y_test)
    
    # Save model and scaler
    save_trained_model(model, scaler, model_dir)
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
