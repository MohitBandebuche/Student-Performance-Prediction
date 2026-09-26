import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

from data_preprocessing import load_data, clean_data, separate_features_target, preprocess_and_split
from eda import plot_feature_importance

def evaluate_model(model, X_test, y_test, model_name):
    \"\"\"Evaluate model performance and return metrics.\"\"\"
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    return {
        'Model': model_name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }

def train_and_evaluate():
    # 1. Load and prepare data
    df = load_data('../data/student_performance.csv')
    df_clean = clean_data(df)
    X, y = separate_features_target(df_clean)
    
    # Preprocess and split
    # This also saves the scaler and encoder in the models directory
    X_train, X_test, y_train, y_test = preprocess_and_split(X, y)
    
    # 2. Define Models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
    }
    
    results = []
    
    # 3. Train and Evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        
    # 4. Display Results Comparison Table
    results_df = pd.DataFrame(results)
    print("\\nModel Evaluation Comparison:")
    print(results_df.to_string(index=False))
    
    # 5. Extract Feature Importance from Random Forest
    rf_model = models['Random Forest']
    importances = rf_model.feature_importances_
    
    # Sort features by importance
    feature_names = X_train.columns
    # Save a plot for reference
    plot_feature_importance(importances, feature_names, save_path='../models/feature_importance.png')
    
    # 6. Save the best model (Random Forest is typically the best among these)
    best_model_path = '../models/random_forest_model.pkl'
    joblib.dump(rf_model, best_model_path)
    print(f"\\nBest model (Random Forest) saved to {best_model_path}")

if __name__ == '__main__':
    train_and_evaluate()
