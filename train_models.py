"""
Model Training Script
Trains all ML models and saves them for use in the application
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.preprocessing import LoanDataPreprocessor
from utils.fraud_detection import FraudDetector
from utils.approval_prediction import LoanApprovalPredictor

def train_all_models():
    """Train and save all models"""
    
    print("=" * 60)
    print("SMART LOAN DECISION SYSTEM - Model Training")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = LoanDataPreprocessor()
    
    # Load and preprocess data
    print("\n[1/5] Loading and preprocessing data...")
    df = preprocessor.load_data('data/raw/loan_data.csv')
    print(f"   ✓ Loaded {len(df)} records")
    
    # Prepare features
    X, y = preprocessor.prepare_features(df)
    print(f"   ✓ Prepared {X.shape[1]} features")
    
    # Save processed data
    processed_df = X.copy()
    processed_df['Loan_Status'] = y
    processed_df.to_csv('data/processed/cleaned_loan_data.csv', index=False)
    print("   ✓ Saved processed data")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✓ Split data: {len(X_train)} train, {len(X_test)} test")
    
    # Train Fraud Detection Model
    print("\n[2/5] Training Fraud Detection Model...")
    fraud_detector = FraudDetector(contamination=0.1)
    fraud_detector.train(X_train)
    
    # Save fraud detection model and feature columns
    os.makedirs('models', exist_ok=True)
    joblib.dump(fraud_detector, 'models/fraud_detection_model.pkl')
    print("   ✓ Fraud detection model trained and saved")
    
    # Train Loan Approval Model
    print("\n[3/5] Training Loan Approval Models (Ensemble)...")
    approval_predictor = LoanApprovalPredictor()
    approval_predictor.train(X_train, y_train)
    
    # Evaluate
    evaluation = approval_predictor.evaluate(X_test, y_test)
    print(f"   ✓ Model Accuracy: {evaluation['accuracy']:.2%}")
    
    # Save approval model
    approval_predictor.save_models('models/loan_approval_model.pkl')
    print("   ✓ Loan approval models trained and saved")
    
    # Display feature importance
    print("\n[4/5] Top 10 Important Features:")
    top_features = approval_predictor.feature_importance.head(10)
    for idx, row in top_features.iterrows():
        print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")
    
    # Save preprocessor
    print("\n[5/5] Saving preprocessor...")
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("   ✓ Preprocessor saved")
    
    # Save scaler
    X_scaled = preprocessor.scale_features(X_train, fit=True)
    joblib.dump(preprocessor.scaler, 'models/scaler.pkl')
    print("   ✓ Scaler saved")
    
    print("\n" + "=" * 60)
    print("✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("=" * 60)
    print("\nModel files saved in 'models/' directory:")
    print("  - fraud_detection_model.pkl")
    print("  - loan_approval_model.pkl")
    print("  - preprocessor.pkl")
    print("  - scaler.pkl")
    print("\n✨ Ready to run the Streamlit application!")
    print("=" * 60)

if __name__ == "__main__":
    train_all_models()