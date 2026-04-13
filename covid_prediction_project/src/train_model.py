from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_logistic_model(X_train, y_train):
    """Trains a Logistic Regression model using GridSearchCV for hyperparameter tuning."""
    param_grid = {
        'solver': ['liblinear', 'lbfgs'],
        'C': [0.01, 0.1, 1, 10],
        'max_iter': [100, 200, 500]
    }
    
    lr = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    print("Logistic Regression model trained with GridSearchCV.")
    return best_model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    
    return accuracy, conf_matrix, class_report
