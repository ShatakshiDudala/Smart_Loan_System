"""
Setup script for Streamlit Cloud deployment
Automatically trains models if they don't exist or are incompatible
"""

import os
import sys
import subprocess

def check_and_train_models():
    """Check if models exist and are compatible, train if needed"""
    
    models_dir = 'models'
    required_models = [
        'fraud_detection_model.pkl',
        'loan_approval_model.pkl',
        'preprocessor.pkl',
        'scaler.pkl'
    ]
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        print("Models directory not found. Creating and training models...")
        os.makedirs(models_dir, exist_ok=True)
        train_models()
        return
    
    # Check if all required model files exist
    missing_models = []
    for model_file in required_models:
        if not os.path.exists(os.path.join(models_dir, model_file)):
            missing_models.append(model_file)
    
    if missing_models:
        print(f"Missing models: {missing_models}")
        print("Training models...")
        train_models()
        return
    
    # Try to load models to check compatibility
    try:
        import joblib
        import warnings
        warnings.filterwarnings('ignore')
        
        # Try loading one model
        joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
        print("✓ Models loaded successfully")
        
    except Exception as e:
        print(f"Models incompatible (trained with different Python/scikit-learn version): {str(e)}")
        print("Retraining models for current environment...")
        train_models()

def train_models():
    """Train all models"""
    try:
        # Import and run training
        print("=" * 60)
        print("TRAINING MODELS FOR STREAMLIT CLOUD")
        print("=" * 60)
        
        # Run train_models.py
        result = subprocess.run([sys.executable, 'train_models.py'], 
                              capture_output=True, 
                              text=True,
                              timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✓ Models trained successfully!")
            print(result.stdout)
        else:
            print("✗ Training failed!")
            print(result.stderr)
            raise Exception("Model training failed")
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    check_and_train_models()