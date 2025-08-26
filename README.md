# CODSOFT-TASK-2

# CodSoft Machine Learning Internship  

This repository contains the solutions to three machine learning projects completed during the *CodSoft Internship*. Each task applies ML techniques to solve real-world problems such as fraud detection, churn prediction, and spam classification.  

---

# Credit Card Fraud Detection Model

A comprehensive machine learning solution for detecting fraudulent credit card transactions using multiple algorithms including Logistic Regression, Decision Trees, and Random Forest.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Requirements](#dataset-requirements)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)

## 🎯 Overview

This project implements a complete fraud detection pipeline that:
- Loads and analyzes credit card transaction data
- Preprocesses data with advanced cleaning techniques
- Trains multiple machine learning models
- Evaluates and compares model performance
- Generates comprehensive visualizations
- Provides detailed performance reports

## ✨ Features

### Data Processing
- **Automatic Data Loading**: Reads CSV files with intelligent column detection
- **Missing Value Handling**: Multiple strategies (median, mean, drop)
- **Data Balancing**: Oversampling techniques for imbalanced datasets
- **Feature Scaling**: Standardization for optimal model performance
- **Categorical Encoding**: Automatic handling of text-based features

### Machine Learning Models
- **Logistic Regression**: Linear baseline with high interpretability
- **Decision Tree**: Rule-based classification with clear decision paths
- **Random Forest**: Ensemble method for robust performance

### Evaluation & Visualization
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visual Analysis**: ROC curves, Precision-Recall curves, Performance heatmaps
- **Feature Importance**: Identifies most significant transaction features
- **Confusion matrices**: Detailed classification breakdowns

## 📦 Requirements

### Python Version
- Python 3.7 or higher

### Required Libraries
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 🚀 Installation

1. **Clone or download the project files**
```bash
# Download the fraud_detection.py file
```

2. **Install required packages**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

3. **Prepare your dataset**
```bash
# Place your fraudtest.csv file in the same directory as the Python script
```

## 📊 Dataset Requirements

### File Format
- **File name**: `fraudtest.csv`
- **Format**: CSV (Comma Separated Values)
- **Encoding**: UTF-8 (recommended)

### Dataset Structure
Your dataset should contain:
- **Transaction features**: Amount, time, merchant info, user details, etc.
- **Target variable**: Binary indicator for fraud (0 = legitimate, 1 = fraud)

### Supported Target Column Names
The script automatically detects these target column names:
- `Class`
- `isFraud`
- `fraud`
- `label`
- `target`
- `is_fraud`
- `y`
- `output`

### Example Dataset Structure
```csv
Time,V1,V2,V3,...,V28,Amount,Class
0,1.234,-0.567,2.345,...,-1.234,149.62,0
406,0.345,1.567,-0.234,...,0.789,2.69,0
541,-2.345,3.567,1.234,...,2.345,1.00,1
```

## 💻 Usage

### Basic Usage
```bash
python fraud_detection.py
```

### Manual Target Column Specification
If your target column has a different name, modify the script:
```python
# In the main() function, change this line:
if not fraud_detector.preprocess_data(target_column=None, balance_data=True):

# To:
if not fraud_detector.preprocess_data(target_column='YourColumnName', balance_data=True):
```

### Customization Options

#### Data Balancing
```python
# Enable/disable data balancing
fraud_detector.preprocess_data(balance_data=True)  
```

#### Missing Value Handling
```python
# Choose strategy: 'median', 'mean', or 'drop'
fraud_detector.preprocess_data(handle_missing='median') 
```

## 📈 Model Performance

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: Fraud detection accuracy
- **Recall**: Fraud case capture rate
- **F1-Score**: Balanced performance measure
- **ROC-AUC**: Area under ROC curve

### Expected Performance
Typical performance ranges for fraud detection:
- **Accuracy**: 85-99%
- **Precision**: 70-95%
- **Recall**: 60-90%
- **F1-Score**: 65-92%

## 📁 Output Files

### Generated Files
1. **`fraud_detection_results.txt`**: Detailed performance report
2. **Performance plots**: Interactive visualizations (displayed on screen)

### Results Structure
```
fraud_detection_results.txt
├── Dataset Information
├── Model Performance Summary
├── Detailed Results for Logistic Regression
├── Detailed Results for Decision Tree
└── Detailed Results for Random Forest
```

## 🔧 Troubleshooting

### Common Issues

#### 1. File Not Found Error
```
FileNotFoundError: [Errno 2] No such file or directory: 'fraudtest.csv'
```
**Solution**: Ensure `fraudtest.csv` is in the same directory as the Python script.

#### 2. Target Column Not Detected
```
Target column 'None' not found!
```
**Solution**: Check your column names and specify manually:
```python
fraud_detector.preprocess_data(target_column='YourTargetColumnName')
```

#### 3. Memory Issues
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce dataset size or use data sampling:
```python
# Add this after loading data
fraud_detector.df = fraud_detector.df.sample(n=10000, random_state=42)
```

#### 4. Import Errors
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**: Install missing packages:
```bash
pip install scikit-learn
```

### Data Quality Issues

#### Missing Values
- Script automatically handles missing values
- Options: median imputation, mean imputation, or row deletion

#### Imbalanced Data
- Automatic detection of class imbalance
- Oversampling applied to balance training data

#### Invalid Data Types
- Automatic conversion of categorical variables
- Binary encoding for target variables

## 🔍 Technical Details

### Algorithm Specifications

#### Logistic Regression
```python
LogisticRegression(
    random_state=42,
    max_iter=1000,
    C=1.0
)
```

#### Decision Tree
```python
DecisionTreeClassifier(
    random_state=42,
    max_depth=5,
    min_samples_split=20
)
```

#### Random Forest
```python
RandomForestClassifier(
    random_state=42,
    n_estimators=50,
    max_depth=5,
    min_samples_split=20
)
```

### Data Preprocessing Pipeline
1. **Data Loading**: CSV file reading with error handling
2. **Data Cleaning**: Remove empty rows and invalid values
3. **Missing Value Imputation**: Statistical imputation methods
4. **Categorical Encoding**: Label encoding for text features
5. **Feature Scaling**: StandardScaler for numerical features
6. **Data Splitting**: 80-20 train-test split with stratification
7. **Class Balancing**: Random oversampling of minority class

### Performance Evaluation
- **Cross-validation**: 5-fold cross-validation for robust estimates
- **Multiple metrics**: Comprehensive evaluation beyond accuracy
- **Visual analysis**: ROC curves and confusion matrices
- **Statistical significance**: Proper train-test separation

---

**Note**: This fraud detection model is designed for educational and research purposes. For production use in financial institutions, additional security measures, regulatory compliance, and extensive testing are required.
