# Student Performance Prediction & Analysis System

## 1. Project Overview
The Student Performance Prediction & Analysis System is an end-to-end data science project designed to analyze student academic and lifestyle data, identify key factors influencing academic success, and predict a student's final score using machine learning.

## 2. Problem Statement
Academic institutions often struggle to identify students who are at risk of poor performance early enough to provide effective interventions. By analyzing various behavioral, academic, and lifestyle metrics, we can build a predictive model to estimate final scores and classify performance.

## 3. Objectives
* Understand the correlation between study hours, attendance, sleep, internet usage, and final grades.
* Perform comprehensive Exploratory Data Analysis (EDA) on student records.
* Clean and preprocess realistic dataset containing missing values and outliers.
* Train and evaluate multiple Regression Models.
* Deploy the final machine learning model via an interactive Streamlit Web Application.

## 4. Features
* **Interactive Dashboard:** View key metrics and dataset previews.
* **Exploratory Data Analysis:** Visualize distributions and correlations between variables.
* **Model Performance Comparison:** Evaluate Linear Regression, Decision Tree, and Random Forest models.
* **Real-time Prediction:** Input student metrics to instantly predict final academic score and performance category (Excellent, Good, Average, Needs Improvement).

## 5. Dataset Description
The dataset (`student_performance.csv`) consists of 1,000+ realistic records generated for this project. 
Key columns include:
* `student_id`: Unique identifier
* `age`, `gender`: Demographics
* `study_hours`, `attendance`, `extracurricular_hours`: Engagement metrics
* `previous_score`, `assignment_score`, `midterm_score`: Academic history
* `sleep_hours`, `internet_usage`: Lifestyle metrics
* `parental_education`: Socio-economic metric
* `final_score`: Target variable to predict

## 6. Technologies Used
* **Python 3**
* **Pandas & NumPy:** Data manipulation and numerical computations
* **Matplotlib & Seaborn:** Data visualization
* **Scikit-learn:** Machine Learning (Preprocessing, Modeling, Evaluation)
* **Joblib:** Model persistence
* **Streamlit:** Interactive web application deployment
* **Jupyter Notebook:** Interactive data exploration

## 7. Project Architecture
```
Student_Performance_Prediction/
│
├── data/
│   └── student_performance.csv       # Generated dataset
│
├── notebooks/
│   └── student_performance_analysis.ipynb # Interactive EDA & Modeling
│
├── src/
│   ├── generate_data.py              # Script to generate realistic data
│   ├── data_preprocessing.py         # Data cleaning and scaling
│   ├── eda.py                        # Visualization functions
│   ├── train_model.py                # Model training and evaluation
│   └── prediction.py                 # Interface for making new predictions
│
├── models/                           # Directory for saved models (pkl files)
├── app.py                            # Streamlit web application
├── requirements.txt                  # Python dependencies
└── README.md
```

## 8. Data Preprocessing
* Handled missing values (Median imputation for numerical, Mode imputation for categorical).
* Removed duplicate records.
* Encoded categorical variables using `OneHotEncoder`.
* Scaled numerical features using `StandardScaler`.

## 9. Machine Learning Models
Three models were evaluated:
1. **Linear Regression:** Baseline model.
2. **Decision Tree Regressor:** Captures non-linear relationships.
3. **Random Forest Regressor:** An ensemble method that provided the highest accuracy and robustness.

## 10. Evaluation Metrics
Models were evaluated using:
* **MAE (Mean Absolute Error)**
* **MSE (Mean Squared Error)**
* **RMSE (Root Mean Squared Error)**
* **R² (R-squared Score)**

## 11. How to Run the Project

**Step 1: Clone or setup the project directory**
Ensure you have Python 3.9+ installed.

**Step 2: Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Generate the dataset**
```bash
cd src
python generate_data.py
cd ..
```

**Step 5: Train the models**
```bash
cd src
python train_model.py
cd ..
```
*This will train the models, save the best Random Forest model to the `models/` directory, and save the feature importance chart.*

**Step 6: Run the Streamlit Application**
```bash
streamlit run app.py
```

## 12. Interview Guide
If you are explaining this project in a placement interview:
1. **Start with the Goal:** "I built a machine learning pipeline that predicts a student's final score based on their lifestyle and academic history."
2. **Explain the Data:** Highlight how you handled missing values and scaled data to ensure model convergence.
3. **Justify Model Selection:** Mention you compared Linear Regression against ensemble methods like Random Forest, and explain *why* Random Forest performed better (e.g., handling non-linear interactions between sleep, study hours, and grades).
4. **Demonstrate Impact:** Show the Streamlit app. Explain how an institution could use this dashboard to flag "Needs Improvement" students early in the semester based on their attendance and midterm scores.

## 13. Future Enhancements
* Deploy the application to AWS/GCP or Streamlit Cloud.
* Integrate Deep Learning models (e.g., simple neural networks).
* Incorporate actual anonymized data from an educational institution.

---
*Developed for Data Science Portfolio & Academic Excellence*
