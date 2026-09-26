import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os

def load_data(filepath):
    \"\"\"Load dataset from CSV file.\"\"\"
    return pd.read_csv(filepath)

def clean_data(df):
    \"\"\"Perform basic data cleaning: handle duplicates and missing values.\"\"\"
    df_cleaned = df.copy()
    
    # 1. Remove duplicate records
    df_cleaned = df_cleaned.drop_duplicates()
    
    # 2. Handle missing values
    # For numerical columns, fill with the median
    num_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        if df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            
    # For categorical columns, fill with the mode
    cat_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
            
    # Optional: drop student_id as it has no predictive power
    if 'student_id' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop('student_id', axis=1)
        
    return df_cleaned

def separate_features_target(df, target_col='final_score'):
    \"\"\"Separate the dataset into features (X) and target variable (y).\"\"\"
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

def preprocess_and_split(X, y, test_size=0.2, random_state=42, models_dir='../models/'):
    \"\"\"
    Split data, encode categorical features, scale numerical features, 
    and save the preprocessors.
    \"\"\"
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Identify numerical and categorical columns
    num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # 1. Encode categorical variables using OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat_encoded = encoder.fit_transform(X_train[cat_cols])
    X_test_cat_encoded = encoder.transform(X_test[cat_cols])
    
    # Get new column names for categorical features
    cat_feature_names = encoder.get_feature_names_out(cat_cols)
    
    # 2. Scale numerical variables using StandardScaler
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train[num_cols])
    X_test_num_scaled = scaler.transform(X_test[num_cols])
    
    # Combine back into DataFrames
    X_train_processed = pd.DataFrame(
        np.hstack((X_train_num_scaled, X_train_cat_encoded)), 
        columns=num_cols + list(cat_feature_names),
        index=X_train.index
    )
    X_test_processed = pd.DataFrame(
        np.hstack((X_test_num_scaled, X_test_cat_encoded)), 
        columns=num_cols + list(cat_feature_names),
        index=X_test.index
    )
    
    # Save the scaler and encoder for future predictions
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(encoder, os.path.join(models_dir, 'encoder.pkl'))
    
    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == '__main__':
    # Test the preprocessing module
    df = load_data('../data/student_performance.csv')
    df_clean = clean_data(df)
    X, y = separate_features_target(df_clean)
    X_train, X_test, y_train, y_test = preprocess_and_split(X, y)
    print("Preprocessing completed successfully!")
    print(f"X_train shape: {X_train.shape}")
