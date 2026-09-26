import pandas as pd
import numpy as np
import os

def generate_student_data(num_records=1000, output_path='../data/student_performance.csv'):
    np.random.seed(42)
    
    # 1. Generate core features
    student_id = np.arange(1, num_records + 1)
    age = np.random.randint(18, 26, size=num_records)
    gender = np.random.choice(['Male', 'Female', 'Other'], size=num_records, p=[0.48, 0.48, 0.04])
    study_hours = np.random.uniform(0, 15, size=num_records)
    attendance = np.random.uniform(50, 100, size=num_records)
    previous_score = np.random.uniform(40, 100, size=num_records)
    assignment_score = np.random.uniform(40, 100, size=num_records)
    midterm_score = np.random.uniform(40, 100, size=num_records)
    sleep_hours = np.random.uniform(4, 10, size=num_records)
    internet_usage = np.random.uniform(0, 10, size=num_records)
    extracurricular_hours = np.random.uniform(0, 20, size=num_records)
    parental_education = np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], size=num_records, p=[0.4, 0.4, 0.15, 0.05])
    
    # 2. Define relationships for target variable (final_score)
    # The final score should be a realistic function of other variables with some noise
    # Positive correlation: study_hours, attendance, previous_score, assignment_score, midterm_score
    # Negative correlation: extreme internet_usage
    # Non-linear/small effect: sleep_hours (optimum around 7-8), extracurricular (optimum around 5-10)
    
    base_score = 10
    score_study = study_hours * 1.2
    score_attendance = (attendance - 50) * 0.4
    score_previous = previous_score * 0.2
    score_assignment = assignment_score * 0.15
    score_midterm = midterm_score * 0.25
    
    # Sleep penalty: penalty if sleep < 6 or sleep > 9
    sleep_penalty = -2 * np.abs(sleep_hours - 7.5)
    
    # Internet penalty
    internet_penalty = -1 * internet_usage
    
    noise = np.random.normal(0, 3, size=num_records)
    
    final_score = (base_score + score_study + score_attendance + score_previous + 
                   score_assignment + score_midterm + sleep_penalty + internet_penalty + noise)
    
    # Clip final scores to be within 0-100 range
    final_score = np.clip(final_score, 0, 100)
    
    # 3. Create DataFrame
    df = pd.DataFrame({
        'student_id': student_id,
        'age': age,
        'gender': gender,
        'study_hours': study_hours,
        'attendance': attendance,
        'previous_score': previous_score,
        'assignment_score': assignment_score,
        'midterm_score': midterm_score,
        'sleep_hours': sleep_hours,
        'internet_usage': internet_usage,
        'extracurricular_hours': extracurricular_hours,
        'parental_education': parental_education,
        'final_score': final_score
    })
    
    # 4. Inject intentional missing values and duplicates for data cleaning demonstration
    # Introduce NaN in 'study_hours', 'attendance', 'previous_score'
    nan_indices = np.random.choice(df.index, size=int(num_records * 0.05), replace=False)
    df.loc[nan_indices[:15], 'study_hours'] = np.nan
    df.loc[nan_indices[15:30], 'attendance'] = np.nan
    df.loc[nan_indices[30:45], 'previous_score'] = np.nan
    
    # Introduce some duplicates
    duplicates = df.sample(n=20, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 5. Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records and saved to {output_path}")

if __name__ == '__main__':
    generate_student_data(1050, '../data/student_performance.csv')
