import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_col='covid_positive'):
    """Preprocesses the data: Splits into features/target, trains-test split, and scales."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Data preprocessed: {X_train_scaled.shape}, {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
