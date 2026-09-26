import pandas as pd
import numpy as np
import joblib
import os

class StudentPerformancePredictor:
    def __init__(self, models_dir='models/'):
        \"\"\"Initialize and load the trained models and preprocessors.\"\"\"
        self.scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        self.encoder = joblib.load(os.path.join(models_dir, 'encoder.pkl'))
        self.model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
        
        # Get categorical and numerical columns based on encoder/scaler expected inputs
        self.cat_cols = self.encoder.feature_names_in_
        self.num_cols = self.scaler.feature_names_in_
        
    def predict(self, student_data):
        \"\"\"
        Predict the final score for a student.
        
        Parameters:
        student_data (dict): A dictionary containing all features of a single student.
        
        Returns:
        float: Predicted final score.
        str: Classification category.
        \"\"\"
        # Convert input dict to DataFrame
        df = pd.DataFrame([student_data])
        
        # Ensure all columns are present
        for col in self.num_cols:
            if col not in df.columns:
                df[col] = 0.0
        for col in self.cat_cols:
            if col not in df.columns:
                df[col] = "Unknown"
        
        # Preprocess Categorical Features
        cat_encoded = self.encoder.transform(df[self.cat_cols])
        cat_feature_names = self.encoder.get_feature_names_out(self.cat_cols)
        
        # Preprocess Numerical Features
        num_scaled = self.scaler.transform(df[self.num_cols])
        
        # Combine processed features
        X_processed = pd.DataFrame(
            np.hstack((num_scaled, cat_encoded)), 
            columns=list(self.num_cols) + list(cat_feature_names)
        )
        
        # Ensure order of columns matches what the model was trained on
        X_processed = X_processed[self.model.feature_names_in_]
        
        # Make Prediction
        predicted_score = self.model.predict(X_processed)[0]
        
        # Classify the score
        category = self.classify_score(predicted_score)
        
        return predicted_score, category

    def classify_score(self, score):
        \"\"\"Classify the numerical score into a category.\"\"\"
        if score >= 85:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 50:
            return "Average"
        else:
            return "Needs Improvement"

if __name__ == '__main__':
    # Test Prediction Module
    # Provide the models_dir correctly depending on where the script is run from
    predictor = StudentPerformancePredictor(models_dir='../models/')
    
    sample_student = {
        'age': 20,
        'gender': 'Female',
        'study_hours': 8.5,
        'attendance': 92.0,
        'previous_score': 85.0,
        'assignment_score': 88.0,
        'midterm_score': 90.0,
        'sleep_hours': 7.5,
        'internet_usage': 3.0,
        'extracurricular_hours': 5.0,
        'parental_education': 'Bachelors'
    }
    
    score, category = predictor.predict(sample_student)
    print(f"Predicted Final Score: {score:.2f}")
    print(f"Category: {category}")
