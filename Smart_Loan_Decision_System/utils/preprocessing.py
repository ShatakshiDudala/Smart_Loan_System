"""
Data Preprocessing Module for Smart Loan Decision System
Handles data cleaning, encoding, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class LoanDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Load raw loan data"""
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        df = df.copy()
        
        # Fill numerical missing values with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical variables"""
        df = df.copy()
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                          'Self_Employed', 'Property_Area']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def create_features(self, df):
        """Create new features for better predictions"""
        df = df.copy()
        
        # Total Income
        df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        
        # Monthly Income
        df['Monthly_Income'] = df['Total_Income']
        
        # Annual Income
        df['Annual_Income'] = df['Total_Income'] * 12
        
        # Loan to Annual Income Ratio
        df['Loan_to_Income_Ratio'] = (df['LoanAmount'] * 1000) / df['Annual_Income']
        
        # Debt-to-Income Ratio (Monthly)
        monthly_emi = self.calculate_emi(df['LoanAmount'] * 1000, 9.0, df['Loan_Amount_Term'])
        df['DTI'] = monthly_emi / df['Monthly_Income']
        
        # Income per dependent
        dependents_numeric = df['Dependents'].replace({'3+': '3'}).astype(float) if df['Dependents'].dtype == 'object' else df['Dependents']
        df['Income_Per_Person'] = df['Total_Income'] / (dependents_numeric + 1)
        
        # Loan Amount per month
        df['Monthly_EMI'] = monthly_emi
        
        # Income Stability Score (higher if co-applicant exists)
        df['Income_Stability'] = np.where(df['CoapplicantIncome'] > 0, 1.2, 1.0)
        
        # Credit Score Category (300-850 range)
        df['Credit_Category'] = pd.cut(df['Credit_Score'], 
                                       bins=[0, 580, 670, 740, 800, 850],
                                       labels=[1, 2, 3, 4, 5])  # 1=Poor, 5=Excellent
        df['Credit_Category'] = df['Credit_Category'].astype(float)
        
        return df
    
    def calculate_emi(self, principal, annual_rate, tenure_months):
        """Calculate EMI"""
        principal = np.asarray(principal)
        tenure_months = np.asarray(tenure_months)
        monthly_rate = annual_rate / (12 * 100)
        
        if monthly_rate == 0:
            return principal / tenure_months
        
        emi = principal * monthly_rate * ((1 + monthly_rate) ** tenure_months) / \
              (((1 + monthly_rate) ** tenure_months) - 1)
        
        return emi
    
    def prepare_features(self, df, target_col='Loan_Status', fit=True):
        """Complete preprocessing pipeline"""
        df = df.copy()
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Create engineered features
        df = self.create_features(df)
        
        # Define expected column order
        feature_columns = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
            'Credit_Score', 'Property_Area', 'Total_Income', 'Monthly_Income',
            'Annual_Income', 'Loan_to_Income_Ratio', 'DTI', 'Income_Per_Person',
            'Monthly_EMI', 'Income_Stability', 'Credit_Category'
        ]
        
        # Separate features and target
        if target_col in df.columns:
            # Encode target variable
            if fit:
                self.label_encoders[target_col] = LabelEncoder()
                y = self.label_encoders[target_col].fit_transform(df[target_col])
            else:
                y = self.label_encoders[target_col].transform(df[target_col])
            
            X = df[feature_columns]
        else:
            y = None
            X = df[feature_columns]
        
        return X, y
    
    def scale_features(self, X, fit=True):
        """Scale numerical features"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def preprocess_single_application(self, application_data):
        """Preprocess a single loan application"""
        df = pd.DataFrame([application_data])
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=False)
        
        # Create features
        df = self.create_features(df)
        
        # Ensure correct column order matching training data
        expected_columns = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
            'Credit_Score', 'Property_Area', 'Total_Income', 'Monthly_Income',
            'Annual_Income', 'Loan_to_Income_Ratio', 'DTI', 'Income_Per_Person',
            'Monthly_EMI', 'Income_Stability', 'Credit_Category'
        ]
        
        # Reorder columns to match training
        df = df[expected_columns]
        
        return df