import matplotlib.pyplot.subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def set_style():
    \"\"\"Set visualization style.\"\"\"
    sns.set_theme(style=\"whitegrid\")
    plt.rcParams['figure.figsize'] = (10, 6)

def plot_distribution(df, column, title, xlabel, ylabel, save_path=None):
    \"\"\"Plot histogram and KDE for a continuous variable.\"\"\"
    set_style()
    plt.figure()
    sns.histplot(df[column], kde=True, bins=30, color='skyblue')
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_scatter(df, x_col, y_col, title, xlabel, ylabel, hue=None, save_path=None):
    \"\"\"Plot a scatter plot to show relationships between two variables.\"\"\"
    set_style()
    plt.figure()
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, alpha=0.7)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if hue:
        plt.legend(title=hue)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_correlation_heatmap(df, save_path=None):
    \"\"\"Plot correlation heatmap for numeric features.\"\"\"
    set_style()
    plt.figure(figsize=(12, 10))
    # Select only numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=\".2f\", linewidths=0.5)
    plt.title("Correlation Heatmap", fontsize=16)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_boxplot(df, x_col, y_col, title, xlabel, ylabel, save_path=None):
    \"\"\"Plot a boxplot for a categorical vs continuous variable.\"\"\"
    set_style()
    plt.figure()
    sns.boxplot(data=df, x=x_col, y=y_col, palette='Set2')
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_feature_importance(importances, feature_names, title="Feature Importances", save_path=None):
    \"\"\"Plot a horizontal bar chart for feature importances.\"\"\"
    set_style()
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], color='mediumseagreen', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title(title, fontsize=14)
    plt.xlabel('Relative Importance', fontsize=12)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
