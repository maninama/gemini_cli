import pandas as pd
import os

def load_data(file_path):
    """Loads the dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    
    data = pd.read_csv(file_path)
    print(f"Dataset loaded: {data.shape}")
    return data
