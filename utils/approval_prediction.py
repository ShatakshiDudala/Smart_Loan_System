"""
Loan Approval Prediction Module
Uses ensemble methods for accurate loan approval predictions
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

class LoanApprovalPredictor:
    def __init__(self):
        """Initialize ensemble of models"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        self.feature_importance = None
        self.feature_names = None
        
    def train(self, X, y):
        """Train all models in the ensemble"""
        self.feature_names = X.columns.tolist()
        
        # Train each model
        for name, model in self.models.items():
            model.fit(X, y)
        
        # Get feature importance from Random Forest
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.models['random_forest'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict approval probability using ensemble voting
        Returns probability of approval (0-1)
        """
        # Ensure correct feature order
        if self.feature_names is not None:
            if isinstance(X, pd.DataFrame):
                X = X[self.feature_names]
        
        # Convert to numpy array to avoid sklearn feature name validation
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        probabilities = []
        
        for model in self.models.values():
            prob = model.predict_proba(X_array)[:, 1]  # Probability of class 1 (approved)
            probabilities.append(prob)
        
        # Average predictions from all models
        ensemble_prob = np.mean(probabilities, axis=0)
        
        return ensemble_prob
    
    def predict(self, X, threshold=0.5):
        """
        Predict loan approval
        Returns: prediction (0/1), probability (0-1)
        """
        probability = self.predict_proba(X)
        prediction = (probability >= threshold).astype(int)
        
        return prediction, probability
    
    def get_approval_factors(self, application_data, probability):
        """
        Get factors affecting approval decision
        """
        factors = []
        
        # Credit Score (300-850 range)
        if 'Credit_Score' in application_data.columns:
            credit_score = application_data['Credit_Score'].values[0]
            if credit_score >= 800:
                factors.append(f"✅ Excellent credit score ({credit_score})")
            elif credit_score >= 740:
                factors.append(f"✅ Very good credit score ({credit_score})")
            elif credit_score >= 670:
                factors.append(f"✅ Good credit score ({credit_score})")
            elif credit_score >= 580:
                factors.append(f"⚠️ Fair credit score ({credit_score})")
            else:
                factors.append(f"❌ Poor credit score ({credit_score})")
        
        # Income level (realistic thresholds)
        if 'Total_Income' in application_data.columns:
            total_income = application_data['Total_Income'].values[0]
            if total_income >= 100000:
                factors.append(f"✅ Excellent income level (₹{total_income:,}/month)")
            elif total_income >= 60000:
                factors.append(f"✅ Very good income level (₹{total_income:,}/month)")
            elif total_income >= 40000:
                factors.append(f"✅ Good income level (₹{total_income:,}/month)")
            elif total_income >= 25000:
                factors.append(f"⚠️ Moderate income level (₹{total_income:,}/month)")
            else:
                factors.append(f"⚠️ Low income level (₹{total_income:,}/month)")
        
        # Loan to Income Ratio
        if 'Loan_to_Income_Ratio' in application_data.columns:
            ratio = application_data['Loan_to_Income_Ratio'].values[0]
            if ratio < 1.0:
                factors.append(f"✅ Excellent affordability (loan is {ratio:.1f}x annual income)")
            elif ratio < 2.0:
                factors.append(f"✅ Good affordability (loan is {ratio:.1f}x annual income)")
            elif ratio < 3.0:
                factors.append(f"⚠️ Moderate affordability (loan is {ratio:.1f}x annual income)")
            else:
                factors.append(f"❌ Low affordability (loan is {ratio:.1f}x annual income)")
        
        # DTI ratio
        if 'DTI' in application_data.columns:
            dti = application_data['DTI'].values[0]
            dti_percent = dti * 100
            if dti < 0.30:
                factors.append(f"✅ Excellent DTI ratio ({dti_percent:.1f}% of income)")
            elif dti < 0.40:
                factors.append(f"✅ Good DTI ratio ({dti_percent:.1f}% of income)")
            elif dti < 0.50:
                factors.append(f"⚠️ Moderate DTI ratio ({dti_percent:.1f}% of income)")
            else:
                factors.append(f"❌ High DTI ratio ({dti_percent:.1f}% of income)")
        
        # Income Stability (Dual income bonus)
        if 'CoapplicantIncome' in application_data.columns:
            coapplicant = application_data['CoapplicantIncome'].values[0]
            if coapplicant > 0:
                factors.append(f"✅ Dual income household (+₹{coapplicant:,})")
        
        # Education
        if 'Education' in application_data.columns:
            if application_data['Education'].values[0] == 1:  # Graduate
                factors.append("✅ Graduate education")
        
        # Employment
        if 'Self_Employed' in application_data.columns:
            if application_data['Self_Employed'].values[0] == 0:
                factors.append("✅ Salaried employment")
        
        # Property area
        if 'Property_Area' in application_data.columns:
            prop_area = application_data['Property_Area'].values[0]
            if prop_area == 2:  # Urban
                factors.append("✅ Urban property location")
        
        return factors
    
    def get_model_confidence(self, probability):
        """
        Calculate model confidence in prediction
        Returns: confidence_score (0-100), confidence_level (Low/Medium/High)
        """
        # Confidence is based on how far probability is from decision boundary (0.5)
        distance_from_boundary = abs(probability - 0.5)
        confidence_score = distance_from_boundary * 200  # Scale to 0-100
        
        if confidence_score >= 70:
            confidence_level = "High"
        elif confidence_score >= 40:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        return confidence_score, confidence_level
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions, probabilities = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'report': report
        }
    
    def save_models(self, filepath):
        """Save all trained models"""
        joblib.dump(self.models, filepath)
    
    def load_models(self, filepath):
        """Load trained models"""
        self.models = joblib.load(filepath)
        return self