import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell("# Student Performance Analysis and Prediction\\nThis notebook covers the exploratory data analysis, data cleaning, preprocessing, and machine learning model training for the student performance dataset."),
        nbf.v4.new_markdown_cell("## 1. Import Libraries"),
        nbf.v4.new_code_cell("import pandas as pd\\nimport numpy as np\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns\\nimport warnings\\nwarnings.filterwarnings('ignore')\\n\\n%matplotlib inline"),
        nbf.v4.new_markdown_cell("## 2. Load the Dataset\\nLet's load the generated student data and display basic statistics."),
        nbf.v4.new_code_cell("df = pd.read_csv('../data/student_performance.csv')\\ndisplay(df.head())\\nprint(f'Dataset Shape: {df.shape}')"),
        nbf.v4.new_markdown_cell("### Data Types and Summary Statistics"),
        nbf.v4.new_code_cell("df.info()"),
        nbf.v4.new_code_cell("df.describe()"),
        nbf.v4.new_markdown_cell("## 3. Data Cleaning\\nWe will identify and handle missing values, as well as remove any duplicate records."),
        nbf.v4.new_code_cell("print('Missing Values before cleaning:')\\nprint(df.isnull().sum())\\n\\nprint(f'\\nDuplicates before cleaning: {df.duplicated().sum()}')"),
        nbf.v4.new_code_cell("# Remove duplicates\\ndf_clean = df.drop_duplicates()\\n\\n# Impute numerical missing values with Median\\nnum_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns\\nfor col in num_cols:\\n    df_clean[col].fillna(df_clean[col].median(), inplace=True)\\n\\nprint('Missing Values after cleaning:')\\nprint(df_clean.isnull().sum())"),
        nbf.v4.new_markdown_cell("## 4. Exploratory Data Analysis (EDA)\\nAnalyzing relationships between features and the target variable (`final_score`)."),
        nbf.v4.new_code_cell("plt.figure(figsize=(8,5))\\nsns.histplot(df_clean['final_score'], kde=True, color='blue')\\nplt.title('Distribution of Final Scores')\\nplt.show()"),
        nbf.v4.new_code_cell("plt.figure(figsize=(8,5))\\nsns.scatterplot(data=df_clean, x='study_hours', y='final_score', hue='gender', alpha=0.7)\\nplt.title('Study Hours vs Final Score')\\nplt.show()"),
        nbf.v4.new_code_cell("plt.figure(figsize=(10,8))\\nnumeric_df = df_clean.select_dtypes(include=[np.number])\\nsns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')\\nplt.title('Correlation Heatmap')\\nplt.show()"),
        nbf.v4.new_markdown_cell("## 5. Machine Learning Modeling\\nWe will treat `final_score` as a regression target. We'll split the data, encode categorical variables, and scale numerical variables."),
        nbf.v4.new_code_cell("from sklearn.model_selection import train_test_split\\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder\\nfrom sklearn.compose import ColumnTransformer\\nfrom sklearn.pipeline import Pipeline\\nfrom sklearn.ensemble import RandomForestRegressor\\nfrom sklearn.linear_model import LinearRegression\\nfrom sklearn.tree import DecisionTreeRegressor\\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score"),
        nbf.v4.new_code_cell("X = df_clean.drop(['final_score', 'student_id'], axis=1)\\ny = df_clean['final_score']\\n\\n# Identify columns\\nnum_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()\\ncat_features = X.select_dtypes(include=['object']).columns.tolist()\\n\\n# Split data\\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"),
        nbf.v4.new_code_cell("# Build preprocessing pipeline\\npreprocessor = ColumnTransformer(\\n    transformers=[\\n        ('num', StandardScaler(), num_features),\\n        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)\\n    ]\\n)\\n\\n# Define models\\nmodels = {\\n    'Linear Regression': LinearRegression(),\\n    'Decision Tree': DecisionTreeRegressor(random_state=42),\\n    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100)\\n}"),
        nbf.v4.new_code_cell("results = []\\nfor name, model in models.items():\\n    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])\\n    pipeline.fit(X_train, y_train)\\n    preds = pipeline.predict(X_test)\\n    results.append({\\n        'Model': name,\\n        'MAE': mean_absolute_error(y_test, preds),\\n        'MSE': mean_squared_error(y_test, preds),\\n        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),\\n        'R2': r2_score(y_test, preds)\\n    })\\n\\nresults_df = pd.DataFrame(results)\\ndisplay(results_df)"),
        nbf.v4.new_markdown_cell("### Feature Importance (Random Forest)\\nFeature importance represents predictive contribution within the trained model and does not necessarily establish real-world causation."),
        nbf.v4.new_code_cell("rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42, n_estimators=100))])\\nrf_pipeline.fit(X_train, y_train)\\n\\n# Extract feature names\\ncat_encoder = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']\\nencoded_cat_features = cat_encoder.get_feature_names_out(cat_features)\\nall_features = num_features + list(encoded_cat_features)\\n\\n# Plot\\nimportances = rf_pipeline.named_steps['model'].feature_importances_\\nindices = np.argsort(importances)\\n\\nplt.figure(figsize=(10, 8))\\nplt.barh(range(len(indices)), importances[indices], color='mediumseagreen')\\nplt.yticks(range(len(indices)), [all_features[i] for i in indices])\\nplt.title('Feature Importances - Random Forest')\\nplt.xlabel('Relative Importance')\\nplt.show()")
    ]
    
    nb['cells'] = cells
    
    import os
    os.makedirs('../notebooks', exist_ok=True)
    with open('../notebooks/student_performance_analysis.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_notebook()
