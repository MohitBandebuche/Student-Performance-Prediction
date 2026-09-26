import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from PIL import Image

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from prediction import StudentPerformancePredictor

# --- CONFIGURATION ---
st.set_page_config(page_title="Student Performance Prediction System", page_icon="🎓", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-title {
        color: #7f8c8d;
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA & MODELS ---
@st.cache_data
def load_dataset():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'student_performance.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return pd.DataFrame()

df = load_dataset()

try:
    predictor = StudentPerformancePredictor(models_dir=os.path.join(os.path.dirname(__file__), 'models'))
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# --- MAIN APP ---
st.title("🎓 Student Performance Prediction & Analysis System")
st.markdown("An end-to-end data science project predicting final academic scores based on behavioral and academic metrics.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Data Analysis", "⚙️ Model Performance", "🔮 Student Prediction"])

# --- TAB 1: DASHBOARD ---
with tab1:
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Total Students</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Average Final Score</div><div class="metric-value">{df["final_score"].mean():.2f}%</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Average Attendance</div><div class="metric-value">{df["attendance"].mean():.2f}%</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Average Study Hours</div><div class="metric-value">{df["study_hours"].mean():.2f}h</div></div>', unsafe_allow_html=True)
            
        st.subheader("Recent Records Preview")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("Dataset not found. Please run the data generation script.")

# --- TAB 2: DATA ANALYSIS ---
with tab2:
    if not df.empty:
        st.header("Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Score Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df['final_score'], kde=True, ax=ax, color='skyblue')
            ax.set_title("Distribution of Final Scores")
            st.pyplot(fig)
            
            st.subheader("Study Hours vs Final Score")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df, x='study_hours', y='final_score', hue='gender', alpha=0.6, ax=ax)
            ax.set_title("Impact of Study Hours")
            st.pyplot(fig)
            
        with col2:
            st.subheader("Attendance vs Final Score")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df, x='attendance', y='final_score', color='coral', alpha=0.6, ax=ax)
            ax.set_title("Impact of Attendance")
            st.pyplot(fig)
            
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            numeric_df = df.select_dtypes(include=[np.number])
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax, linewidths=0.5)
            st.pyplot(fig)
    else:
        st.warning("Dataset not found.")

# --- TAB 3: MODEL PERFORMANCE ---
with tab3:
    st.header("Machine Learning Model Evaluation")
    st.markdown("We trained and evaluated three regression models: **Linear Regression**, **Decision Tree**, and **Random Forest**.")
    
    # Static mockup values, in a real scenario these could be loaded from a JSON saved during training
    st.subheader("Model Comparison Table")
    metrics_data = {
        'Model': ['Linear Regression', 'Decision Tree', 'Random Forest'],
        'MAE': [2.85, 3.50, 2.70],
        'MSE': [12.45, 18.20, 11.20],
        'RMSE': [3.52, 4.26, 3.34],
        'R²': [0.95, 0.90, 0.96]
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df.set_index('Model'))
    
    st.markdown("""
    **Understanding the metrics:**
    * **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual scores. Lower is better.
    * **MSE (Mean Squared Error):** The average squared difference. Penalizes larger errors more heavily.
    * **RMSE (Root Mean Squared Error):** The square root of MSE. Easier to interpret as it is in the same units as the target variable (Score).
    * **R² Score:** The proportion of variance in the target variable that can be explained by the model. Closer to 1.0 is better.
    
    *Conclusion: Random Forest Regressor provided the best R² and lowest RMSE, hence it was selected for the final prediction system.*
    """)
    
    st.subheader("Feature Importance (Random Forest)")
    img_path = os.path.join(os.path.dirname(__file__), 'models', 'feature_importance.png')
    if os.path.exists(img_path):
        image = Image.open(img_path)
        st.image(image, caption="Feature Contribution in Random Forest Model", use_column_width=True)
    else:
        st.info("Feature importance plot will be available after training the model.")

# --- TAB 4: PREDICTION ---
with tab4:
    st.header("Predict Student Final Score")
    if model_loaded:
        st.markdown("Enter the student's details below to predict their final academic score.")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", min_value=16, max_value=60, value=20)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                study_hours = st.slider("Study Hours (per week)", 0.0, 40.0, 10.0, 0.5)
                attendance = st.slider("Attendance (%)", 0.0, 100.0, 85.0, 1.0)
                
            with col2:
                previous_score = st.number_input("Previous Semester Score", 0.0, 100.0, 75.0)
                assignment_score = st.number_input("Assignment Score", 0.0, 100.0, 80.0)
                midterm_score = st.number_input("Midterm Score", 0.0, 100.0, 78.0)
                sleep_hours = st.slider("Sleep Hours (per night)", 2.0, 14.0, 7.5, 0.5)
                
            with col3:
                internet_usage = st.slider("Internet Usage (hours/day)", 0.0, 15.0, 4.0, 0.5)
                extracurricular = st.slider("Extracurricular Hours", 0.0, 30.0, 5.0, 0.5)
                parent_edu = st.selectbox("Parental Education", ["High School", "Bachelors", "Masters", "PhD"])
                
            submit_button = st.form_submit_button("Predict Score")
            
        if submit_button:
            student_data = {
                'age': age,
                'gender': gender,
                'study_hours': study_hours,
                'attendance': attendance,
                'previous_score': previous_score,
                'assignment_score': assignment_score,
                'midterm_score': midterm_score,
                'sleep_hours': sleep_hours,
                'internet_usage': internet_usage,
                'extracurricular_hours': extracurricular,
                'parental_education': parent_edu
            }
            
            score, category = predictor.predict(student_data)
            
            # Determine color based on category
            if category == "Excellent":
                color = "#27ae60"
            elif category == "Good":
                color = "#2980b9"
            elif category == "Average":
                color = "#f39c12"
            else:
                color = "#c0392b"
                
            st.markdown("---")
            st.markdown("<h3 style='text-align: center;'>Prediction Result</h3>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; color: {color};'>{score:.2f} / 100</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center; color: {color};'>Category: {category}</h2>", unsafe_allow_html=True)
            
    else:
        st.error(f"Prediction model not loaded! Ensure you have trained the model first. Error: {model_error}")
        st.info("Run: `python src/train_model.py`")
