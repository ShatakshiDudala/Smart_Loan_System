"""
Fraud Detection Module using Isolation Forest
Detects anomalous loan applications that may be fraudulent
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

class FraudDetector:
    def __init__(self, contamination=0.1):
        """
        Initialize Fraud Detector
        contamination: Expected proportion of outliers (fraud cases)
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.feature_columns = None
        
    def train(self, X):
        """Train the fraud detection model"""
        self.feature_columns = X.columns.tolist()
        self.model.fit(X)
        return self
    
    def predict(self, X):
        """
        Predict fraud probability
        Returns: fraud_score (0-100), is_fraudulent (bool)
        """
        # Convert to DataFrame if needed and ensure correct column order
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_columns)
        
        # Reorder columns to match training if feature_columns is set
        if self.feature_columns is not None:
            # Only select columns that exist in both
            X = X[self.feature_columns]
        
        # Convert to numpy array to avoid sklearn feature name validation
        X_array = X.values
        
        # Get anomaly scores using numpy array
        scores = self.model.decision_function(X_array)
        predictions = self.model.predict(X_array)
        
        # Convert to fraud probability (0-100)
        # Normalize scores to 0-100 range
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score != 0:
            fraud_scores = 100 * (1 - (scores - min_score) / (max_score - min_score))
        else:
            fraud_scores = np.zeros(len(scores))
        
        # Predictions: -1 for fraud, 1 for legitimate
        is_fraud = predictions == -1
        
        return fraud_scores, is_fraud
    
    def get_fraud_indicators(self, application_data):
        """
        Identify specific fraud indicators in the application
        """
        indicators = []
        
        # High DTI ratio
        if 'DTI' in application_data.columns and application_data['DTI'].values[0] > 0.6:
            indicators.append("⚠️ Extremely high Debt-to-Income ratio (>60%)")
        
        # Unusually high loan amount relative to income
        if 'Loan_to_Income_Ratio' in application_data.columns:
            ratio = application_data['Loan_to_Income_Ratio'].values[0]
            if ratio > 5:
                indicators.append(f"⚠️ Loan amount is {ratio:.1f}x annual income (very high)")
        
        # Poor credit score with high loan amount
        if 'Credit_Score' in application_data.columns and 'LoanAmount' in application_data.columns:
            credit_score = application_data['Credit_Score'].values[0]
            loan_amount = application_data['LoanAmount'].values[0]
            if credit_score < 600 and loan_amount > 500:
                indicators.append(f"⚠️ Poor credit score ({credit_score}) with high loan request (₹{loan_amount * 1000:,})")
        
        # Very low income with high loan request
        if 'Total_Income' in application_data.columns and 'LoanAmount' in application_data.columns:
            income = application_data['Total_Income'].values[0]
            loan = application_data['LoanAmount'].values[0]
            if income < 25000 and loan > 500:
                indicators.append(f"⚠️ Low income (₹{income:,}) with high loan request (₹{loan * 1000:,})")
        
        # Suspicious credit score (too perfect or unrealistic)
        if 'Credit_Score' in application_data.columns:
            credit_score = application_data['Credit_Score'].values[0]
            if credit_score < 300 or credit_score > 850:
                indicators.append(f"⚠️ Invalid credit score ({credit_score}) - should be 300-850")
        
        if not indicators:
            indicators.append("✅ No major fraud indicators detected")
        
        return indicators