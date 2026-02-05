"""
Explainability Module using SHAP
Provides interpretable explanations for loan decisions
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

class LoanExplainer:
    def __init__(self, model, X_train):
        """
        Initialize SHAP explainer
        model: Trained model (use Random Forest from ensemble)
        X_train: Training data for background samples
        """
        self.model = model
        self.explainer = shap.TreeExplainer(model, X_train)
        self.feature_names = X_train.columns.tolist()
        
    def explain_prediction(self, X):
        """
        Generate SHAP values for prediction explanation
        """
        # Convert DataFrame to numpy array for SHAP
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        shap_values = self.explainer.shap_values(X_array)
        
        # For binary classification, use positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        return shap_values
    
    def get_top_features(self, shap_values, X, n_features=5):
        """
        Get top features affecting the decision
        """
        # Get absolute SHAP values
        abs_shap = np.abs(shap_values[0])
        
        # Get indices of top features
        top_indices = np.argsort(abs_shap)[-n_features:][::-1]
        
        # Get feature values
        if isinstance(X, pd.DataFrame):
            X_values = X.iloc[0].values
        else:
            X_values = X[0] if len(X.shape) > 1 else X
        
        # Create explanation
        explanations = []
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            feature_value = X_values[idx]
            shap_value = shap_values[0][idx]
            
            # Determine impact direction
            impact = "increases" if shap_value > 0 else "decreases"
            
            # Create readable feature names
            readable_names = {
                'Credit_History': 'Credit History',
                'Total_Income': 'Total Income',
                'DTI': 'Debt-to-Income Ratio',
                'LoanAmount': 'Loan Amount',
                'EMI_to_Income': 'EMI to Income Ratio',
                'Education': 'Education Level',
                'Property_Area': 'Property Location',
                'Married': 'Marital Status',
                'Self_Employed': 'Employment Type',
                'ApplicantIncome': 'Applicant Income'
            }
            
            readable_name = readable_names.get(feature_name, feature_name)
            
            explanations.append({
                'feature': readable_name,
                'value': feature_value,
                'impact': impact,
                'shap_value': abs(shap_value)
            })
        
        return explanations
    
    def create_waterfall_plot(self, shap_values, X):
        """Create SHAP waterfall plot"""
        try:
            # Get values for plotting
            if isinstance(X, pd.DataFrame):
                X_values = X.iloc[0].values
            else:
                X_values = X[0] if len(X.shape) > 1 else X
                
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
                    data=X_values,
                    feature_names=self.feature_names
                ),
                max_display=10,
                show=False
            )
            plt.tight_layout()
            return fig
        except Exception as e:
            return None
    
    def create_force_plot(self, shap_values, X):
        """Create SHAP force plot"""
        try:
            # Get values for plotting
            if isinstance(X, pd.DataFrame):
                X_values = X.iloc[0].values
            else:
                X_values = X[0] if len(X.shape) > 1 else X
                
            base_value = self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value
            
            fig = shap.force_plot(
                base_value,
                shap_values[0],
                X_values,
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            
            return fig
        except Exception as e:
            return None
    
    def get_decision_explanation(self, shap_values, X, prediction_prob):
        """
        Generate human-readable explanation
        """
        top_features = self.get_top_features(shap_values, X, n_features=3)
        
        if prediction_prob >= 0.5:
            decision = "APPROVED"
            explanation = "The loan application is likely to be **approved** based on the following key factors:"
        else:
            decision = "REJECTED"
            explanation = "The loan application is likely to be **rejected** based on the following key factors:"
        
        explanation += "\n\n"
        
        for i, feature in enumerate(top_features, 1):
            explanation += f"{i}. **{feature['feature']}**: This factor {feature['impact']} the approval probability.\n"
        
        return explanation