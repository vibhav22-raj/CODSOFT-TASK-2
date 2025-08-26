
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

class FraudDetectionModel:
    def __init__(self, csv_file):
        """
        Initialize the fraud detection model with dataset
        """
        self.csv_file = csv_file
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_explore_data(self):
        """
        Load the dataset and perform initial exploration
        """
        print("Loading dataset...")
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"Dataset loaded successfully!")
            print(f"Shape: {self.df.shape}")
            print("\nFirst few rows:")
            print(self.df.head())
            print("\nDataset Info:")
            print(self.df.info())
            print("\nBasic Statistics:")
            print(self.df.describe())
           
            missing_values = self.df.isnull().sum()
            print(f"\nMissing values:\n{missing_values[missing_values > 0]}")
            
            print(f"\nAll columns in dataset: {list(self.df.columns)}")
            
            possible_target_cols = ['Class', 'isFraud', 'fraud', 'label', 'target', 'is_fraud', 'y', 'output']
            target_col = None
            
            for col in possible_target_cols:
                if col in self.df.columns:
                    target_col = col
                    break
            
            if target_col:
                print(f"\nTarget variable '{target_col}' found!")
                print(f"Class distribution:\n{self.df[target_col].value_counts()}")
                
                if self.df[target_col].isnull().sum() > 0:
                    print(f"WARNING: Target column has {self.df[target_col].isnull().sum()} missing values!")
                else:
                    fraud_rate = self.df[target_col].mean()
                    print(f"Fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")
            else:
                print("\nTarget variable not automatically detected.")
                print("Available columns:", list(self.df.columns))
                print("Please specify the target column manually in the preprocess_data() call.")
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
        
        return True
    
    def clean_data(self):
        """
        Clean the dataset by handling missing values and invalid data
        """
        print("\nCleaning data...")
        initial_shape = self.df.shape
        
        self.df = self.df.dropna(how='all')
        print(f"Removed {initial_shape[0] - self.df.shape[0]} rows with all missing values")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        print(f"Data shape after initial cleaning: {self.df.shape}")
        return True
    
    def preprocess_data(self, target_column=None, balance_data=True, handle_missing='median'):
        """
        Preprocess the data for modeling with improved missing value handling
        """
        if self.df is None:
            print("Please load the data first!")
            return False
        
        self.clean_data()
        
        if target_column is None:
            possible_targets = ['Class', 'isFraud', 'fraud', 'label', 'target', 'is_fraud', 'y', 'output']
            for col in possible_targets:
                if col in self.df.columns:
                    target_column = col
                    break
        
        if target_column is None or target_column not in self.df.columns:
            print(f"Target column '{target_column}' not found!")
            print("Available columns:", list(self.df.columns))
            print("Please specify the correct target column name.")
            return False
        
        print(f"Using '{target_column}' as target variable")
        
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=[target_column])
        removed_rows = initial_rows - len(self.df)
        if removed_rows > 0:
            print(f"Removed {removed_rows} rows with missing target values")
        
        y = self.df[target_column]
        X = self.df.drop(columns=[target_column])
        
        if y.dtype == 'object' or len(y.unique()) > 2:
            print("Converting target variable to binary format...")
            
            if y.dtype == 'object':
                fraud_indicators = ['fraud', 'fraudulent', '1', 'yes', 'true', 'positive']
                y = y.astype(str).str.lower().isin(fraud_indicators).astype(int)
            else:
                y = (y != 0).astype(int)
        
        print(f"Final target distribution:\n{pd.Series(y).value_counts()}")
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"Encoding categorical columns: {list(categorical_cols)}")
            le = LabelEncoder()
            for col in categorical_cols:
                
                X[col] = X[col].fillna('Unknown')
                X[col] = le.fit_transform(X[col].astype(str))
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if X[numeric_cols].isnull().sum().sum() > 0:
            print("Handling missing values in numeric columns...")
            if handle_missing == 'median':
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
            elif handle_missing == 'mean':
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
            elif handle_missing == 'drop':
               
                missing_mask = X[numeric_cols].isnull().any(axis=1)
                X = X[~missing_mask]
                y = y[~missing_mask]
                print(f"Dropped {missing_mask.sum()} rows with missing values")
        
        if X.isnull().sum().sum() > 0:
            print("WARNING: Still have missing values. Filling with 0...")
            X = X.fillna(0)
   
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Target distribution: {pd.Series(y).value_counts()}")
        
        try:
            if pd.Series(y).value_counts().min() >= 2:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                print("WARNING: Too few samples in minority class for stratified split. Using regular split.")
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
        except Exception as e:
            print(f"Error in train_test_split: {e}")
            print("Using regular split without stratification...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
     
        if balance_data and pd.Series(self.y_train).value_counts().min() > 0:
            self.balance_training_data()
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Training set class distribution:\n{pd.Series(self.y_train).value_counts()}")
        
        return True
    
    def balance_training_data(self):
        """
        Balance the training data using random oversampling
        """
        if isinstance(self.y_train, np.ndarray):
            y_train_series = pd.Series(self.y_train)
        else:
            y_train_series = self.y_train
        
        train_data = pd.concat([pd.DataFrame(self.X_train), y_train_series], axis=1)
        target_col_name = train_data.columns[-1]
        
        majority_class = train_data[train_data[target_col_name] == 0]
        minority_class = train_data[train_data[target_col_name] == 1]
        
        print(f"Original training distribution - Legitimate: {len(majority_class)}, Fraud: {len(minority_class)}")
        
        if len(minority_class) == 0:
            print("WARNING: No fraud cases in training data. Skipping balancing.")
            return
        
        minority_upsampled = resample(minority_class,
                                    replace=True,
                                    n_samples=len(majority_class),
                                    random_state=42)
        
        balanced_data = pd.concat([majority_class, minority_upsampled])
        
        self.y_train = balanced_data[target_col_name].values
        self.X_train = balanced_data.drop(columns=[target_col_name]).values
        
        print(f"Balanced training distribution - Legitimate: {(self.y_train == 0).sum()}, Fraud: {(self.y_train == 1).sum()}")
    
    def train_models(self):
        """
        Train multiple models for fraud detection
        """
        print("\nTraining models...")
        
        if len(self.X_train) == 0:
            print("ERROR: No training data available!")
            return False
        
        models_config = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=20),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50, max_depth=5, min_samples_split=20)
        }
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            try:
                if name == 'Logistic Regression':
                    model.fit(self.X_train_scaled, self.y_train)
                    y_pred = model.predict(self.X_test_scaled)
                    y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                self.models[name] = model
                self.results[name] = {
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred, zero_division=0),
                    'recall': recall_score(self.y_test, y_pred, zero_division=0),
                    'f1': f1_score(self.y_test, y_pred, zero_division=0),
                    'roc_auc': roc_auc_score(self.y_test, y_pred_proba) if len(np.unique(self.y_test)) > 1 else 0.5
                }
                
                print(f"{name} trained successfully!")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        return len(self.models) > 0
    
    def evaluate_models(self):
        """
        Evaluate all trained models
        """
        if not self.models:
            print("No trained models to evaluate!")
            return
            
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        results_df = pd.DataFrame(self.results).T
        print("\nSummary of Model Performance:")
        print(results_df.round(4))
        
        for name in self.models.keys():
            print(f"\n{'-'*50}")
            print(f"DETAILED RESULTS FOR {name.upper()}")
            print(f"{'-'*50}")
            
            y_pred = self.results[name]['predictions']
            
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred, zero_division=0))
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(self.y_test, y_pred)
            print(cm)
            if cm.shape == (2, 2):
                print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
                print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    def plot_results(self):
        """
        Create visualizations of model performance
        """
        if not self.models:
            print("No trained models to plot!")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        model_names = list(self.models.keys())
        
        performance_data = []
        for metric in metrics:
            performance_data.append([self.results[model][metric] for model in model_names])
        
        performance_df = pd.DataFrame(performance_data, 
                                    columns=model_names, 
                                    index=metrics)
        
        sns.heatmap(performance_df, annot=True, cmap='viridis', 
                   ax=axes[0,0], cbar_kws={'label': 'Score'})
        axes[0,0].set_title('Model Performance Heatmap')
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('Metrics')
        
        axes[0,1].plot([0, 1], [0, 1], 'k--', label='Random')
        for name in self.models.keys():
            if len(np.unique(self.y_test)) > 1:  
                fpr, tpr, _ = roc_curve(self.y_test, self.results[name]['probabilities'])
                auc_score = self.results[name]['roc_auc']
                axes[0,1].plot(fpr, tpr, label=f'{name} (AUC={auc_score:.3f})')
        
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curves')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        for name in self.models.keys():
            if len(np.unique(self.y_test)) > 1:  
                precision, recall, _ = precision_recall_curve(self.y_test, 
                                                            self.results[name]['probabilities'])
                axes[1,0].plot(recall, precision, label=name)
        
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_title('Precision-Recall Curves')
        axes[1,0].legend()
        axes[1,0].grid(True)
       
        if 'Random Forest' in self.models and hasattr(self.X_train, 'columns'):
            rf_model = self.models['Random Forest']
            if hasattr(self.X_train, 'columns'):
                feature_names = self.X_train.columns
            else:
                feature_names = [f'Feature_{i}' for i in range(self.X_train.shape[1])]
                
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[-10:]  
            
            axes[1,1].barh(range(len(indices)), importances[indices])
            axes[1,1].set_yticks(range(len(indices)))
            axes[1,1].set_yticklabels([feature_names[i] for i in indices])
            axes[1,1].set_xlabel('Feature Importance')
            axes[1,1].set_title('Top 10 Feature Importances (Random Forest)')
        else:
            axes[1,1].text(0.5, 0.5, 'Feature importance plot\nnot available', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, filename='fraud_detection_results.txt'):
        """
        Save model results to a text file
        """
        if not self.models:
            print("No results to save!")
            return
            
        with open(filename, 'w') as f:
            f.write("CREDIT CARD FRAUD DETECTION RESULTS\n")
            f.write("="*50 + "\n\n")
            
            f.write("Dataset Information:\n")
            f.write(f"Total samples: {len(self.df)}\n")
            f.write(f"Training samples: {len(self.X_train)}\n")
            f.write(f"Test samples: {len(self.X_test)}\n\n")
            
            f.write("Model Performance Summary:\n")
            results_df = pd.DataFrame(self.results).T
            f.write(results_df.round(4).to_string())
            f.write("\n\n")
            
            for name in self.models.keys():
                f.write(f"\nDetailed Results for {name}:\n")
                f.write("-" * 30 + "\n")
                y_pred = self.results[name]['predictions']
                f.write(classification_report(self.y_test, y_pred, zero_division=0))
                f.write("\n")
        
        print(f"Results saved to {filename}")

def main():
    """
    Main function to run the fraud detection pipeline
    """
    fraud_detector = FraudDetectionModel('fraudtest.csv')
    
    print("Step 1: Loading and exploring data...")
    if not fraud_detector.load_and_explore_data():
        print("Failed to load data. Please check your CSV file.")
        return
  
    print("\nStep 2: Preprocessing data...")
    if not fraud_detector.preprocess_data(target_column=None, balance_data=True, handle_missing='median'):
        print("Data preprocessing failed.")
        print("Please check the target column name and try again.")
        return
  
    print("\nStep 3: Training models...")
    if not fraud_detector.train_models():
        print("Model training failed.")
        return

    print("\nStep 4: Evaluating models...")
    fraud_detector.evaluate_models()
    
    print("\nStep 5: Generating plots...")
    try:
        fraud_detector.plot_results()
    except Exception as e:
        print(f"Error generating plots: {e}")
        print("Continuing with saving results...")

    print("\nStep 6: Saving results...")
    fraud_detector.save_results()
    
    print("\n" + "="*60)
    print("FRAUD DETECTION ANALYSIS COMPLETED!")
    print("="*60)
    print("Check the generated plots and 'fraud_detection_results.txt' for detailed results.")
    print("If you encountered any issues, the script has handled them gracefully.")

if __name__ == "__main__":
    main()