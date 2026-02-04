"""
Smart Loan Decision System - Main Application
A comprehensive ML-based loan approval system with fraud detection and explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import os
import sys
import json
import bcrypt
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from utils.preprocessing import LoanDataPreprocessor
from utils.fraud_detection import FraudDetector
from utils.approval_prediction import LoanApprovalPredictor
from utils.explainability import LoanExplainer
from utils.emi_calculator import EMICalculator
from utils.risk_scoring import RiskScorer

# Page configuration
st.set_page_config(
    page_title="Smart Loan Decision System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1e3a8a;
        --secondary-color: #3b82f6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    .danger-card {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Input styling */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        padding: 0.75rem;
    }
    
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f3f4f6;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Data frame */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'application_history' not in st.session_state:
    st.session_state.application_history = []

# User database (in production, use proper database)
USERS_FILE = 'users.json'

def load_users():
    """Load users from file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def hash_password(password):
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def signup_page():
    """User registration page"""
    st.markdown('<div class="main-header"><h1>🏦 Smart Loan Decision System</h1><p>Sign Up for New Account</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Create Your Account")
        
        with st.form("signup_form"):
            username = st.text_input("Username", placeholder="Enter username")
            email = st.text_input("Email", placeholder="Enter email")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password")
            role = st.selectbox("Role", ["User", "Admin"])
            
            submit = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submit:
                if not username or not email or not password:
                    st.error("❌ All fields are required!")
                elif password != confirm_password:
                    st.error("❌ Passwords do not match!")
                elif len(password) < 6:
                    st.error("❌ Password must be at least 6 characters!")
                else:
                    users = load_users()
                    
                    if username in users:
                        st.error("❌ Username already exists!")
                    else:
                        users[username] = {
                            'email': email,
                            'password': hash_password(password),
                            'role': role,
                            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        save_users(users)
                        st.success("✅ Account created successfully! Please login.")
                        st.balloons()

def login_page():
    """User login page"""
    st.markdown('<div class="main-header"><h1>🏦 Smart Loan Decision System</h1><p>AI-Powered Loan Approval Platform</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Login to Your Account")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col_login, col_signup = st.columns(2)
            
            with col_login:
                submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                users = load_users()
                
                if username in users and verify_password(password, users[username]['password']):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.user_role = users[username]['role']
                    st.success(f"✅ Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password!")
        
        if st.button("Create New Account", use_container_width=True):
            st.session_state.page = 'signup'
            st.rerun()

def logout():
    """Logout user"""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_role = None
    st.rerun()

# Load models
@st.cache_resource
def load_models():
    """Load all trained models"""
    try:
        preprocessor = joblib.load('models/preprocessor.pkl')
        fraud_detector = joblib.load('models/fraud_detection_model.pkl')
        approval_predictor = LoanApprovalPredictor()
        approval_predictor.load_models('models/loan_approval_model.pkl')
        
        # Load training data for SHAP
        train_data = pd.read_csv('data/processed/cleaned_loan_data.csv')
        X_train = train_data.drop('Loan_Status', axis=1)
        
        return preprocessor, fraud_detector, approval_predictor, X_train
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please run 'train_models.py' first to train the models.")
        return None, None, None, None

def home_page():
    """Home/Dashboard page"""
    st.markdown('<div class="main-header"><h1>🏦 Smart Loan Decision System</h1><p>AI-Powered Intelligent Loan Approval Platform</p></div>', unsafe_allow_html=True)
    
    # Welcome message
    st.markdown(f"### Welcome, {st.session_state.username}! 👋")
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>📊</h2>
            <h3>Total Applications</h3>
            <h1>{}</h1>
        </div>
        """.format(len(st.session_state.application_history)), unsafe_allow_html=True)
    
    with col2:
        approved = sum(1 for app in st.session_state.application_history if app.get('decision') == 'Approved')
        st.markdown("""
        <div class="success-card">
            <h2>✅</h2>
            <h3>Approved</h3>
            <h1>{}</h1>
        </div>
        """.format(approved), unsafe_allow_html=True)
    
    with col3:
        rejected = sum(1 for app in st.session_state.application_history if app.get('decision') == 'Rejected')
        st.markdown("""
        <div class="danger-card">
            <h2>❌</h2>
            <h3>Rejected</h3>
            <h1>{}</h1>
        </div>
        """.format(rejected), unsafe_allow_html=True)
    
    with col4:
        fraud_detected = sum(1 for app in st.session_state.application_history if app.get('fraud_detected', False))
        st.markdown("""
        <div class="warning-card">
            <h2>⚠️</h2>
            <h3>Fraud Detected</h3>
            <h1>{}</h1>
        </div>
        """.format(fraud_detected), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Features
    st.markdown("### 🌟 System Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🔍 Fraud Detection</h4>
            <p>Advanced machine learning algorithms detect suspicious applications using Isolation Forest technique.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>🎯 Approval Prediction</h4>
            <p>Ensemble models provide accurate loan approval predictions with confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>📈 Risk Assessment</h4>
            <p>Comprehensive risk scoring based on multiple financial and demographic factors.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🤖 Explainable AI</h4>
            <p>SHAP values provide transparent explanations for every loan decision made.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>💰 EMI Calculator</h4>
            <p>Smart recommendations for loan amount and EMI based on income and affordability.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>🔮 What-If Analysis</h4>
            <p>Test different scenarios to find the optimal loan terms for your situation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats visualization
    if st.session_state.application_history:
        st.markdown("---")
        st.markdown("### 📊 Recent Application Trends")
        
        df_history = pd.DataFrame(st.session_state.application_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Approval rate pie chart
            decision_counts = df_history['decision'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=decision_counts.index,
                values=decision_counts.values,
                hole=0.4,
                marker=dict(colors=['#10b981', '#ef4444', '#f59e0b'])
            )])
            fig.update_layout(
                title="Decision Distribution",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk distribution
            if 'risk_level' in df_history.columns:
                risk_counts = df_history['risk_level'].value_counts()
                fig = go.Figure(data=[go.Bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    marker=dict(color=['#10b981', '#f59e0b', '#ef4444'])
                )])
                fig.update_layout(
                    title="Risk Level Distribution",
                    xaxis_title="Risk Level",
                    yaxis_title="Count",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

def main_app():
    """Main application after login"""
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        st.markdown(f"**Role:** {st.session_state.user_role}")
        st.markdown("---")
        
        selected = option_menu(
            menu_title="Navigation",
            options=["Home", "New Application", "Interest Calculator", "Collateral Assessment", "Application History", "Analytics", "About"],
            icons=["house", "file-earmark-plus", "calculator", "bank", "clock-history", "bar-chart", "info-circle"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#667eea", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#667eea"},
            }
        )
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            logout()
    
    # Load models
    preprocessor, fraud_detector, approval_predictor, X_train = load_models()
    
    if preprocessor is None:
        st.error("⚠️ Models not loaded. Please train models first by running: `python train_models.py`")
        return
    
    # Page routing
    if selected == "Home":
        home_page()
    elif selected == "New Application":
        new_application_page(preprocessor, fraud_detector, approval_predictor, X_train)
    elif selected == "Interest Calculator":
        interest_calculator_page()
    elif selected == "Collateral Assessment":
        collateral_assessment_page()
    elif selected == "Application History":
        application_history_page()
    elif selected == "Analytics":
        analytics_page()
    elif selected == "About":
        about_page()

def new_application_page(preprocessor, fraud_detector, approval_predictor, X_train):
    """New loan application page"""
    st.markdown('<div class="main-header"><h1>📝 New Loan Application</h1><p>Submit Your Loan Application</p></div>', unsafe_allow_html=True)
    
    # Application form
    with st.form("loan_application"):
        st.markdown("### 👤 Personal Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        
        with col2:
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        
        with col3:
            credit_score = st.slider(
                "Credit Score (FICO Scale)",
                min_value=300,
                max_value=850,
                value=700,
                step=10,
                help="300-579: Poor | 580-669: Fair | 670-739: Good | 740-799: Very Good | 800-850: Excellent"
            )
            
            # Display credit category
            if credit_score >= 800:
                st.success("🟢 Excellent Credit")
            elif credit_score >= 740:
                st.info("🟦 Very Good Credit")
            elif credit_score >= 670:
                st.warning("🟡 Good Credit")
            elif credit_score >= 580:
                st.warning("🟠 Fair Credit")
            else:
                st.error("🔴 Poor Credit")
        
        st.markdown("---")
        st.markdown("### 💰 Financial Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            applicant_income = st.number_input(
                "Applicant Monthly Income (₹)", 
                min_value=0, 
                value=50000, 
                step=5000,
                format="%d"
            )
            coapplicant_income = st.number_input(
                "Co-applicant Monthly Income (₹)", 
                min_value=0, 
                value=0, 
                step=5000,
                format="%d"
            )
            
            # Show total income
            total_income = applicant_income + coapplicant_income
            st.metric("💼 Total Monthly Income", f"₹{total_income:,}")
            st.metric("📅 Annual Income", f"₹{total_income * 12:,}")
        
        with col2:
            loan_amount = st.number_input(
                "Loan Amount (₹)",
                min_value=50000,
                max_value=50000000,
                value=500000,
                step=50000,
                format="%d",
                help="Enter the total loan amount you need"
            )
            loan_term = st.selectbox(
                "Loan Tenure", 
                [60, 120, 180, 240, 300, 360, 480],
                index=5,
                format_func=lambda x: f"{x} months ({x//12} years)"
            )
            
            # Show loan to income ratio
            if total_income > 0:
                loan_to_income = loan_amount / (total_income * 12)
                st.metric("📊 Loan to Annual Income Ratio", f"{loan_to_income:.2f}x")
                
                if loan_to_income < 1:
                    st.success("✅ Excellent - Well within capacity")
                elif loan_to_income < 2:
                    st.info("✅ Good - Reasonable amount")
                elif loan_to_income < 3:
                    st.warning("⚠️ Moderate - Consider carefully")
                else:
                    st.error("❌ High - May be challenging")
        
        st.markdown("---")
        
        submit = st.form_submit_button("🚀 Submit Application", use_container_width=True)
        
        if submit:
            # Prepare application data with Credit_Score
            application_data = {
                'Gender': gender,
                'Married': married,
                'Dependents': dependents,
                'Education': education,
                'Self_Employed': self_employed,
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount / 1000,  # Convert to thousands for model
                'Loan_Amount_Term': loan_term,
                'Credit_Score': credit_score,  # Changed from Credit_History
                'Property_Area': property_area
            }
            
            # Process application
            with st.spinner("🔄 Processing your application..."):
                process_application(application_data, preprocessor, fraud_detector, approval_predictor, X_train)
    
    # Download button OUTSIDE the form
    if 'loan_summary' in st.session_state and st.session_state.loan_summary:
        st.markdown("---")
        st.markdown("### 📥 Download Your Loan Summary")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.download_button(
                label="📄 Download Complete Loan Analysis (TXT)",
                data=st.session_state.loan_summary,
                file_name=f"loan_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                type="primary",
                use_container_width=True
            )
            st.caption("💾 Complete summary of your loan application including all calculations and recommendations")

def process_application(application_data, preprocessor, fraud_detector, approval_predictor, X_train):
    """Process loan application through the entire pipeline"""
    
    # Preprocess data
    df_app = preprocessor.preprocess_single_application(application_data)
    
    st.success("✅ Application submitted successfully!")
    st.markdown("---")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Fraud Detection", 
        "🎯 Approval Prediction", 
        "🤖 Explainability", 
        "📊 Risk Assessment",
        "💰 Loan Recommendation"
    ])
    
    # Tab 1: Fraud Detection
    with tab1:
        st.markdown("### 🔍 Fraud Detection Analysis")
        
        fraud_scores, is_fraud = fraud_detector.predict(df_app)
        fraud_score = fraud_scores[0]
        is_fraudulent = is_fraud[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud score gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=fraud_score,
                title={'text': "Fraud Risk Score"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if fraud_score > 70 else "orange" if fraud_score > 40 else "green"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if is_fraudulent:
                st.markdown("""
                <div class="danger-card">
                    <h3>⚠️ FRAUD ALERT</h3>
                    <p>This application shows suspicious patterns and requires manual verification.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-card">
                    <h3>✅ LEGITIMATE</h3>
                    <p>No major fraud indicators detected. Application appears genuine.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Fraud indicators
            st.markdown("#### Fraud Indicators:")
            indicators = fraud_detector.get_fraud_indicators(df_app)
            for indicator in indicators:
                st.markdown(f"- {indicator}")
    
    # Tab 2: Approval Prediction
    with tab2:
        st.markdown("### 🎯 Loan Approval Prediction")
        
        prediction, probability = approval_predictor.predict(df_app)
        approval_prob = probability[0]
        is_approved = prediction[0] == 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Approval probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=approval_prob * 100,
                title={'text': "Approval Probability"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "green" if approval_prob > 0.7 else "orange" if approval_prob > 0.4 else "red"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightcoral"},
                        {'range': [40, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "green", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Model confidence
            confidence_score, confidence_level = approval_predictor.get_model_confidence(approval_prob)
            
            st.metric("Decision", "✅ APPROVED" if is_approved else "❌ REJECTED")
            st.metric("Confidence Level", confidence_level)
            st.metric("Confidence Score", f"{confidence_score:.1f}%")
            
            # Approval factors
            st.markdown("#### Key Factors:")
            factors = approval_predictor.get_approval_factors(df_app, approval_prob)
            for factor in factors[:5]:
                st.markdown(f"- {factor}")
    
    # Tab 3: Explainability
    with tab3:
        st.markdown("### 🤖 AI Decision Explanation (SHAP Analysis)")
        
        try:
            # Create explainer
            rf_model = approval_predictor.models['random_forest']
            explainer = LoanExplainer(rf_model, X_train)
            
            # Get SHAP values
            shap_values = explainer.explain_prediction(df_app)
            
            # Get explanation
            explanation = explainer.get_decision_explanation(shap_values, df_app, approval_prob)
            st.markdown(explanation)
            
            st.markdown("---")
            
            # Top features
            st.markdown("#### Top Features Affecting Decision:")
            top_features = explainer.get_top_features(shap_values, df_app, n_features=5)
            
            feature_df = pd.DataFrame(top_features)
            
            # Create bar chart
            fig = go.Figure(go.Bar(
                x=feature_df['shap_value'],
                y=feature_df['feature'],
                orientation='h',
                marker=dict(
                    color=feature_df['shap_value'],
                    colorscale='RdYlGn',
                    showscale=True
                )
            ))
            fig.update_layout(
                title="Feature Impact on Decision",
                xaxis_title="Impact Score",
                yaxis_title="Feature",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {str(e)}")
    
    # Tab 4: Risk Assessment
    with tab4:
        st.markdown("### 📊 Risk Assessment")
        
        risk_scorer = RiskScorer()
        
        # Prepare data for risk scoring with Credit_Score
        risk_data = {
            'Credit_Score': application_data['Credit_Score'],  # Changed from Credit_History
            'Total_Income': df_app['Total_Income'].values[0],
            'ApplicantIncome': application_data['ApplicantIncome'],
            'CoapplicantIncome': application_data['CoapplicantIncome'],
            'DTI': df_app['DTI'].values[0],
            'Self_Employed': 1 if application_data['Self_Employed'] == 'Yes' else 0,
            'Property_Area': 2 if application_data['Property_Area'] == 'Urban' else 1 if application_data['Property_Area'] == 'Semiurban' else 0,
            'Dependents': application_data['Dependents'],
            'Education': 1 if application_data['Education'] == 'Graduate' else 0,
            'LoanAmount': application_data['LoanAmount']
        }
        
        risk_result = risk_scorer.calculate_risk_score(risk_data)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Risk score display
            risk_color_map = {'green': '#10b981', 'orange': '#f59e0b', 'red': '#ef4444'}
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_result['risk_score'],
                title={'text': "Overall Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_color_map[risk_result['risk_color']]},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 55], 'color': "lightyellow"},
                        {'range': [55, 100], 'color': "lightcoral"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Risk Level", risk_result['risk_level'])
        
        with col2:
            st.markdown("#### Risk Breakdown by Category:")
            
            breakdown_df = pd.DataFrame({
                'Category': list(risk_result['breakdown'].keys()),
                'Score': list(risk_result['breakdown'].values())
            })
            
            fig = go.Figure(go.Bar(
                x=breakdown_df['Score'],
                y=breakdown_df['Category'],
                orientation='h',
                marker=dict(color='#667eea')
            ))
            fig.update_layout(
                xaxis_title="Risk Score",
                yaxis_title="Category",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors
        st.markdown("#### Detailed Risk Factors:")
        risk_factors = risk_scorer.get_risk_factors(risk_data, risk_result['breakdown'])
        
        for factor in risk_factors:
            severity_colors = {
                'low': 'success-card',
                'medium': 'warning-card',
                'high': 'danger-card'
            }
            
            st.markdown(f"""
            <div class="{severity_colors.get(factor['severity'], 'info-card')}">
                <h5>{factor['category']}</h5>
                <p><strong>Status:</strong> {factor['status']}</p>
                <p>{factor['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Improvement suggestions
        st.markdown("---")
        st.markdown("#### 💡 Suggestions for Improvement:")
        suggestions = risk_scorer.get_improvement_suggestions(risk_result['breakdown'], risk_data)
        
        for suggestion in suggestions:
            st.markdown(suggestion)
    
    # Tab 5: Loan Recommendation
    with tab5:
        st.markdown("### 💰 Loan Amount & EMI Recommendation")
        
        emi_calc = EMICalculator()
        
        # Get recommended interest rate using Credit_Score
        total_income = application_data['ApplicantIncome'] + application_data['CoapplicantIncome']
        interest_rate, rate_category = emi_calc.recommend_interest_rate(
            application_data['Credit_Score'],  # Changed from Credit_History
            1 if application_data['Self_Employed'] == 'Yes' else 0,
            total_income
        )
        
        # Display interest rate prominently
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="success-card">
                <h3>📊 Interest Rate</h3>
                <h1>{interest_rate}%</h1>
                <p>Per Annum</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-card">
                <h4>Credit Category</h4>
                <p><strong>{rate_category}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Calculate total interest
            loan_amt = application_data['LoanAmount'] * 1000
            tenure = application_data['Loan_Amount_Term']
            emi = emi_calc.calculate_emi(loan_amt, interest_rate, tenure)
            total_payment = emi * tenure
            total_interest = total_payment - loan_amt
            total_interest_lakhs = total_interest / 100000
            
            st.markdown(f"""
            <div class="warning-card">
                <h4>Total Interest</h4>
                <h2>₹{total_interest_lakhs:.2f}L</h2>
                <p style="font-size:12px;">₹{total_interest:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Get loan recommendation
        recommendation = emi_calc.recommend_safe_loan_amount(
            monthly_income=total_income,
            annual_rate=interest_rate,
            tenure_months=application_data['Loan_Amount_Term'],
            requested_amount=application_data['LoanAmount'] * 1000  # Convert thousands to actual amount
        )
        
        # Display recommendation with Indian currency formatting (Lakhs and Thousands)
        st.markdown("### 📋 Loan Recommendation Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Convert to lakhs/thousands for better readability
        loan_amt_lakhs = recommendation['recommended_amount'] / 100000
        emi_thousands = recommendation['monthly_emi'] / 1000
        total_pay_lakhs = recommendation['total_payment'] / 100000
        
        with col1:
            st.metric(
                "💵 Recommended Amount", 
                f"₹{loan_amt_lakhs:.1f}L",
                help=f"₹{recommendation['recommended_amount']:,.0f}"
            )
        
        with col2:
            st.metric(
                "📅 Monthly EMI", 
                f"₹{emi_thousands:.1f}K",
                help=f"₹{recommendation['monthly_emi']:,.0f}"
            )
        
        with col3:
            st.metric(
                "💰 Total Payment", 
                f"₹{total_pay_lakhs:.2f}L",
                help=f"₹{recommendation['total_payment']:,.0f}"
            )
        
        with col4:
            st.metric("📊 EMI/Income Ratio", f"{recommendation['emi_to_income_ratio']:.1f}%")
        
        # Recommendation message
        rec_type_colors = {
            'Approved as Requested': 'success-card',
            'Approved with Caution': 'warning-card',
            'Reduced Amount Recommended': 'danger-card'
        }
        
        st.markdown(f"""
        <div class="{rec_type_colors.get(recommendation['recommendation_type'], 'info-card')}">
            <h4>{recommendation['recommendation_type']}</h4>
            <p>{recommendation['message']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # DETAILED INTEREST BREAKDOWN WITH CHARTS
        st.markdown("### 📊 Complete Loan & Interest Breakdown")
        
        # Create beautiful pie chart for loan breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Payment Distribution")
            
            # Pie chart showing principal vs interest
            fig = go.Figure(data=[go.Pie(
                labels=['Principal Amount', 'Interest Charged'],
                values=[recommendation['recommended_amount'], recommendation['total_interest']],
                hole=0.4,
                marker=dict(colors=['#10b981', '#f59e0b']),
                textinfo='label+percent+value',
                texttemplate='<b>%{label}</b><br>₹%{value:,.0f}<br>(%{percent})',
            )])
            
            fig.update_layout(
                title=f"Total Payment: ₹{recommendation['total_payment']:,.0f}",
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 💳 Interest Rate Breakdown")
            
            # Show interest rate components
            base_rate = 7.5 if application_data['Credit_Score'] >= 800 else \
                       8.5 if application_data['Credit_Score'] >= 740 else \
                       9.5 if application_data['Credit_Score'] >= 670 else \
                       10.5 if application_data['Credit_Score'] >= 580 else 12.5
            
            rate_adjustments = []
            rate_adjustments.append(('Base Rate', base_rate, '#3b82f6'))
            
            if application_data['Self_Employed'] == 'Yes':
                rate_adjustments.append(('Self-Employed +', 0.5, '#f59e0b'))
            
            if total_income < 30000:
                rate_adjustments.append(('Low Income +', 1.0, '#ef4444'))
            elif total_income < 50000:
                rate_adjustments.append(('Income Adj +', 0.5, '#f59e0b'))
            
            # Bar chart for rate breakdown
            fig = go.Figure()
            
            cumulative = 0
            for label, value, color in rate_adjustments:
                fig.add_trace(go.Bar(
                    x=[value],
                    y=[label],
                    orientation='h',
                    marker=dict(color=color),
                    text=f'{value}%',
                    textposition='auto',
                    name=label
                ))
                cumulative += value
            
            fig.update_layout(
                title=f"Final Rate: {interest_rate}%",
                xaxis_title="Rate (%)",
                height=400,
                showlegend=False,
                barmode='stack'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Monthly breakdown over time
        st.markdown("### 📈 EMI Payment Over Time")
        
        # Create payment trend chart
        months_to_show = min(recommendation['tenure_months'], 60)  # Show max 5 years
        schedule = emi_calc.create_amortization_schedule(
            recommendation['recommended_amount'],
            interest_rate,
            recommendation['tenure_months'],
            num_rows=months_to_show
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Stacked area chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=schedule['Month'],
                y=schedule['Interest'],
                mode='lines',
                name='Interest Payment',
                fill='tozeroy',
                line=dict(color='#f59e0b', width=0),
                fillcolor='rgba(245, 158, 11, 0.5)'
            ))
            
            fig.add_trace(go.Scatter(
                x=schedule['Month'],
                y=schedule['Principal'] + schedule['Interest'],
                mode='lines',
                name='Principal Payment',
                fill='tonexty',
                line=dict(color='#10b981', width=0),
                fillcolor='rgba(16, 185, 129, 0.5)'
            ))
            
            fig.update_layout(
                title=f"Payment Breakdown - First {months_to_show} Months",
                xaxis_title="Month",
                yaxis_title="Amount (₹)",
                height=400,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Key Insights")
            
            st.metric("🏦 Loan Amount", f"₹{recommendation['recommended_amount']:,.0f}")
            st.metric("📅 Loan Tenure", f"{recommendation['tenure_years']:.1f} years")
            st.metric("💰 Monthly EMI", f"₹{recommendation['monthly_emi']:,.2f}")
            st.metric("💸 Total Interest", f"₹{recommendation['total_interest']:,.0f}")
            st.metric("💵 Total Payment", f"₹{recommendation['total_payment']:,.0f}")
            
            # Interest to Principal ratio
            interest_ratio = (recommendation['total_interest'] / recommendation['recommended_amount']) * 100
            st.metric("📈 Interest Ratio", f"{interest_ratio:.1f}%")
        
        st.markdown("---")
        
        # Amortization schedule table
        st.markdown("#### 📅 Detailed Payment Schedule (First Year)")
        
        # Show first 12 months
        schedule_display = schedule.head(12).copy()
        schedule_display['EMI'] = schedule_display['EMI'].apply(lambda x: f"₹{x:,.2f}")
        schedule_display['Principal'] = schedule_display['Principal'].apply(lambda x: f"₹{x:,.2f}")
        schedule_display['Interest'] = schedule_display['Interest'].apply(lambda x: f"₹{x:,.2f}")
        schedule_display['Balance'] = schedule_display['Balance'].apply(lambda x: f"₹{x:,.0f}")
        
        st.dataframe(schedule_display, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Comparison chart: Different tenure options
        st.markdown("### 🔄 Tenure Comparison")
        
        tenure_options = [120, 180, 240, 300, 360]
        tenure_data = []
        
        for tenure_opt in tenure_options:
            emi_opt = emi_calc.calculate_emi(recommendation['recommended_amount'], interest_rate, tenure_opt)
            total_payment_opt = emi_opt * tenure_opt
            total_interest_opt = total_payment_opt - recommendation['recommended_amount']
            
            tenure_data.append({
                'Tenure': f"{tenure_opt//12} years",
                'Monthly EMI': emi_opt,
                'Total Interest': total_interest_opt,
                'Total Payment': total_payment_opt
            })
        
        tenure_df = pd.DataFrame(tenure_data)
        
        # Create comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=tenure_df['Tenure'],
            y=tenure_df['Monthly EMI'],
            name='Monthly EMI',
            marker=dict(color='#667eea'),
            text=tenure_df['Monthly EMI'].apply(lambda x: f"₹{x:,.0f}"),
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            x=tenure_df['Tenure'],
            y=tenure_df['Total Interest'],
            name='Total Interest',
            marker=dict(color='#f59e0b'),
            text=tenure_df['Total Interest'].apply(lambda x: f"₹{x:,.0f}"),
            textposition='auto'
        ))
        
        fig.update_layout(
            title="How Tenure Affects Your Payments",
            xaxis_title="Loan Tenure",
            yaxis_title="Amount (₹)",
            height=400,
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display comparison table
        tenure_df_display = tenure_df.copy()
        tenure_df_display['Monthly EMI'] = tenure_df_display['Monthly EMI'].apply(lambda x: f"₹{x:,.0f}")
        tenure_df_display['Total Interest'] = tenure_df_display['Total Interest'].apply(lambda x: f"₹{x:,.0f}")
        tenure_df_display['Total Payment'] = tenure_df_display['Total Payment'].apply(lambda x: f"₹{x:,.0f}")
        
        st.dataframe(tenure_df_display, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # What-if analysis
        st.markdown("### 🔮 What-If Scenarios")
        
        scenarios = emi_calc.what_if_analysis(
            base_income=total_income,
            base_loan=application_data['LoanAmount'] * 1000,
            annual_rate=interest_rate,
            tenure_months=application_data['Loan_Amount_Term']
        )
        
        # Create scenario comparison chart
        fig = go.Figure()
        
        colors_scenarios = ['#667eea', '#764ba2', '#10b981', '#f59e0b']
        
        for idx, row in scenarios.iterrows():
            fig.add_trace(go.Bar(
                name=row['scenario'],
                x=['Monthly EMI'],
                y=[row['monthly_emi']],
                marker_color=colors_scenarios[idx],
                text=f"₹{row['monthly_emi']:,.0f}<br>{row['emi_percentage']:.1f}%",
                textposition='auto',
                hovertemplate=f"<b>{row['scenario']}</b><br>" +
                             f"Loan: ₹{row['loan_amount']:,.0f}<br>" +
                             f"EMI: ₹{row['monthly_emi']:,.0f}<br>" +
                             f"EMI %: {row['emi_percentage']:.1f}%<br>" +
                             f"Status: {row['affordability']}<extra></extra>"
            ))
        
        fig.update_layout(
            title="EMI Comparison Across Different Scenarios",
            yaxis_title="Monthly EMI (₹)",
            height=400,
            barmode='group',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display scenarios table with color coding
        st.markdown("#### 📋 Scenario Details")
        
        for idx, row in scenarios.iterrows():
            affordability_color = {
                'Safe': 'success-card',
                'Risky': 'warning-card',
                'Very Risky': 'danger-card'
            }
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"**{row['scenario']}**")
            with col2:
                st.markdown(f"Loan: ₹{row['loan_amount']:,.0f}")
            with col3:
                st.markdown(f"EMI: ₹{row['monthly_emi']:,.0f}")
            with col4:
                st.markdown(f"""
                <span style="padding: 4px 8px; border-radius: 4px; background-color: {'#10b981' if row['affordability'] == 'Safe' else '#f59e0b' if row['affordability'] == 'Risky' else '#ef4444'}; color: white;">
                    {row['affordability']}
                </span>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Download button for detailed report
        st.markdown("### 📥 Download Loan Details")
        
        # Create downloadable summary
        summary_text = f"""
LOAN DETAILS SUMMARY
===================

Application Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

APPLICANT INFORMATION
---------------------
Monthly Income: ₹{total_income:,}
Annual Income: ₹{total_income * 12:,}
Credit Score: {application_data['Credit_Score']}

LOAN DETAILS
-----------
Requested Amount: ₹{application_data['LoanAmount'] * 1000:,}
Recommended Amount: ₹{recommendation['recommended_amount']:,}
Interest Rate: {interest_rate}% per annum
Loan Tenure: {recommendation['tenure_years']:.1f} years ({recommendation['tenure_months']} months)

PAYMENT BREAKDOWN
----------------
Monthly EMI: ₹{recommendation['monthly_emi']:,.2f}
Total Payment: ₹{recommendation['total_payment']:,}
Total Interest: ₹{recommendation['total_interest']:,}
EMI to Income Ratio: {recommendation['emi_to_income_ratio']:.1f}%

RECOMMENDATION
-------------
{recommendation['recommendation_type']}
{recommendation['message']}

This is a computer-generated summary and for informational purposes only.
"""
        
        # Store summary in session state for download outside form
        st.session_state['loan_summary'] = summary_text
        
        st.success("✅ Summary generated! Scroll to the very bottom of this page (below all tabs) to download the complete loan analysis.")
    
    # Save to history
    history_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'applicant_income': application_data['ApplicantIncome'],
        'loan_amount': application_data['LoanAmount'],
        'fraud_score': float(fraud_score),
        'fraud_detected': bool(is_fraudulent),
        'approval_probability': float(approval_prob),
        'decision': 'Approved' if is_approved else 'Rejected',
        'risk_score': risk_result['risk_score'],
        'risk_level': risk_result['risk_level'],
        'recommended_emi': recommendation['monthly_emi']
    }
    
    st.session_state.application_history.append(history_entry)
    
    # Success message at the end
    st.success("✅ Application processed successfully! Results displayed above.")

def interest_calculator_page():
    """Interest Calculator - Standalone page for calculating loan costs"""
    st.markdown('<div class="main-header"><h1>🧮 Interest Calculator</h1><p>Calculate Your Loan Costs</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 💡 How to Use This Calculator
    Use this tool to understand how much interest you'll pay for different loan amounts and tenures.
    All amounts are in **Indian Lakhs (₹)** for easier understanding.
    """)
    
    st.markdown("---")
    
    # Calculator inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Loan Details")
        
        loan_amount_lakhs = st.number_input(
            "Loan Amount (in Lakhs ₹)",
            min_value=1.0,
            max_value=500.0,
            value=10.0,
            step=0.5,
            help="Enter loan amount in lakhs. E.g., 10 = ₹10,00,000"
        )
        
        tenure_years = st.selectbox(
            "Loan Tenure (Years)",
            options=[5, 10, 15, 20, 25, 30],
            index=3,
            help="Select how many years to repay the loan"
        )
        
        credit_score_calc = st.slider(
            "Your Credit Score",
            min_value=300,
            max_value=850,
            value=700,
            step=10,
            help="Your FICO credit score determines your interest rate"
        )
    
    with col2:
        st.markdown("#### 📅 Loan Timeline")
        
        loan_start_date = st.date_input(
            "Loan Disbursement Date",
            value=datetime.now(),
            help="When will the loan be disbursed?"
        )
        
        # Calculate end date
        loan_end_date = loan_start_date + timedelta(days=tenure_years * 365)
        
        st.info(f"""
        **Loan Period:**
        - **Start Date:** {loan_start_date.strftime('%d %B %Y')}
        - **End Date:** {loan_end_date.strftime('%d %B %Y')}
        - **Total Duration:** {tenure_years} years ({tenure_years * 12} months)
        - **Final Payment Due:** {loan_end_date.strftime('%d %B %Y')}
        """)
        
        st.warning(f"⏰ **Deadline to repay full loan:** {loan_end_date.strftime('%d %B %Y')}")
    
    st.markdown("---")
    
    # Calculate interest rate based on credit score
    emi_calc = EMICalculator()
    interest_rate, rate_category = emi_calc.recommend_interest_rate(credit_score_calc, 0, 50000)
    
    # Convert lakhs to actual amount
    loan_amount = loan_amount_lakhs * 100000  # Convert lakhs to rupees
    tenure_months = tenure_years * 12
    
    # Calculate EMI and totals
    monthly_emi = emi_calc.calculate_emi(loan_amount, interest_rate, tenure_months)
    total_payment = monthly_emi * tenure_months
    total_interest = total_payment - loan_amount
    
    # Display results in colorful cards
    st.markdown("### 💰 Loan Cost Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <h4>📊 Interest Rate</h4>
            <h1>{interest_rate}%</h1>
            <p>{rate_category}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="success-card">
            <h4>💵 Loan Amount</h4>
            <h1>₹{loan_amount_lakhs:.1f}L</h1>
            <p>₹{loan_amount:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="warning-card">
            <h4>📅 Monthly EMI</h4>
            <h1>₹{monthly_emi/1000:.1f}K</h1>
            <p>₹{monthly_emi:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="danger-card">
            <h4>💸 Total Interest</h4>
            <h1>₹{total_interest/100000:.2f}L</h1>
            <p>₹{total_interest:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed breakdown table
    st.markdown("### 📊 Complete Cost Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Payment summary table
        summary_data = {
            'Component': ['Principal Amount', 'Total Interest', 'Total Payment', 'Monthly EMI', 'Interest %'],
            'Amount (₹)': [
                f"₹{loan_amount_lakhs:.1f}L (₹{loan_amount:,.0f})",
                f"₹{total_interest/100000:.2f}L (₹{total_interest:,.0f})",
                f"₹{total_payment/100000:.2f}L (₹{total_payment:,.0f})",
                f"₹{monthly_emi/1000:.1f}K (₹{monthly_emi:,.0f})",
                f"{(total_interest/loan_amount)*100:.1f}%"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Principal', 'Interest'],
            values=[loan_amount, total_interest],
            hole=0.5,
            marker=dict(colors=['#10b981', '#ef4444']),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>₹%{value:,.0f}<extra></extra>'
        )])
        
        fig_pie.update_layout(
            title=f"Total Cost: ₹{total_payment/100000:.2f}L",
            height=300
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Interest comparison table for different loan amounts
    st.markdown("### 📋 Interest Comparison Table")
    st.markdown("Compare interest costs for different loan amounts with your selected tenure and credit score")
    
    # Generate comparison data
    loan_amounts_lakhs = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50]
    comparison_data = []
    
    for amt_lakhs in loan_amounts_lakhs:
        amt = amt_lakhs * 100000
        emi = emi_calc.calculate_emi(amt, interest_rate, tenure_months)
        total_pay = emi * tenure_months
        total_int = total_pay - amt
        interest_pct = (total_int/amt)*100
        
        comparison_data.append({
            'Loan Amount': f"₹{amt_lakhs}L",
            'Monthly EMI': f"₹{emi/1000:.1f}K",
            'Total Interest': f"₹{total_int/100000:.2f}L",
            'Total Payment': f"₹{total_pay/100000:.2f}L",
            'Interest %': f"{interest_pct:.1f}%",
            '_interest_pct_numeric': interest_pct  # Hidden numeric column for styling
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display comparison table (without gradient to avoid string conversion errors)
    st.dataframe(
        comparison_df[['Loan Amount', 'Monthly EMI', 'Total Interest', 'Total Payment', 'Interest %']],
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    st.markdown("---")
    
    # Tenure comparison
    st.markdown("### ⏰ How Tenure Affects Your Loan")
    
    tenures = [5, 10, 15, 20, 25, 30]
    tenure_comparison = []
    
    for years in tenures:
        months = years * 12
        emi = emi_calc.calculate_emi(loan_amount, interest_rate, months)
        total_pay = emi * months
        total_int = total_pay - loan_amount
        
        tenure_comparison.append({
            'Tenure': f"{years} years",
            'Monthly EMI': f"₹{emi/1000:.1f}K",
            'Total Interest': f"₹{total_int/100000:.2f}L",
            'Total Payment': f"₹{total_pay/100000:.2f}L"
        })
    
    tenure_df = pd.DataFrame(tenure_comparison)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(tenure_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Chart showing tenure impact
        fig_tenure = go.Figure()
        
        emi_values = [emi_calc.calculate_emi(loan_amount, interest_rate, y*12) for y in tenures]
        interest_values = [(emi_calc.calculate_emi(loan_amount, interest_rate, y*12) * y * 12) - loan_amount for y in tenures]
        
        fig_tenure.add_trace(go.Scatter(
            x=tenures,
            y=[e/1000 for e in emi_values],
            name='Monthly EMI (₹K)',
            mode='lines+markers',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10)
        ))
        
        fig_tenure.add_trace(go.Scatter(
            x=tenures,
            y=[i/100000 for i in interest_values],
            name='Total Interest (₹L)',
            mode='lines+markers',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=10),
            yaxis='y2'
        ))
        
        fig_tenure.update_layout(
            title='EMI vs Interest Across Tenures',
            xaxis_title='Tenure (Years)',
            yaxis=dict(title='Monthly EMI (₹ Thousands)'),
            yaxis2=dict(title='Total Interest (₹ Lakhs)', overlaying='y', side='right'),
            height=350,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_tenure, use_container_width=True)
    
    st.markdown("---")
    
    # Key insights
    st.markdown("### 💡 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Interest Burden:**
        - You'll pay ₹{total_interest/100000:.2f}L in interest
        - That's {(total_interest/loan_amount)*100:.1f}% of the loan amount
        - Almost {total_interest/monthly_emi:.0f} months of EMI is just interest!
        """)
    
    with col2:
        # Calculate for shorter tenure
        shorter_tenure = max(5, tenure_years - 5)
        shorter_emi = emi_calc.calculate_emi(loan_amount, interest_rate, shorter_tenure * 12)
        shorter_total = shorter_emi * shorter_tenure * 12
        shorter_interest = shorter_total - loan_amount
        savings = total_interest - shorter_interest
        
        st.success(f"""
        **Save Money:**
        - Reduce tenure to {shorter_tenure} years
        - Save ₹{savings/100000:.2f}L in interest
        - EMI increases to ₹{shorter_emi/1000:.1f}K
        """)
    
    with col3:
        # Credit score impact
        better_score = min(850, credit_score_calc + 100)
        better_rate, _ = emi_calc.recommend_interest_rate(better_score, 0, 50000)
        better_emi = emi_calc.calculate_emi(loan_amount, better_rate, tenure_months)
        better_total = better_emi * tenure_months
        better_interest = better_total - loan_amount
        score_savings = total_interest - better_interest
        
        st.warning(f"""
        **Better Credit Score:**
        - Improve score to {better_score}
        - Get {better_rate}% rate (from {interest_rate}%)
        - Save ₹{score_savings/100000:.2f}L over loan life
        """)

def collateral_assessment_page():
    """Collateral/Property Assessment page"""
    st.markdown('<div class="main-header"><h1>🏦 Collateral Assessment</h1><p>Secure Your Loan with Property</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 💼 What is Collateral?
    Collateral is an asset (like property, land, or vehicle) that you pledge to the bank as security for your loan.
    If you're unable to repay the loan, the bank can legally take possession of the collateral.
    """)
    
    st.markdown("---")
    
    # Property details input
    st.markdown("### 🏠 Property Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        property_type = st.selectbox(
            "Property Type",
            ["Residential House", "Apartment", "Commercial Property", "Land/Plot", "Agricultural Land"],
            help="Type of property you're offering as collateral"
        )
        
        property_value_lakhs = st.number_input(
            "Property Market Value (₹ Lakhs)",
            min_value=5.0,
            max_value=1000.0,
            value=50.0,
            step=5.0,
            help="Current market value of your property"
        )
        
        property_age = st.number_input(
            "Property Age (Years)",
            min_value=0,
            max_value=100,
            value=5,
            help="How old is the property?"
        )
    
    with col2:
        property_location = st.selectbox(
            "Property Location",
            ["Metro City", "Tier 1 City", "Tier 2 City", "Town", "Rural Area"],
            help="Where is the property located?"
        )
        
        ownership_status = st.selectbox(
            "Ownership Status",
            ["Fully Owned", "Under Mortgage", "Ancestral Property", "Joint Ownership"],
            help="Current ownership status"
        )
        
        property_condition = st.selectbox(
            "Property Condition",
            ["Excellent", "Good", "Average", "Needs Renovation"],
            help="Current condition of the property"
        )
    
    st.markdown("---")
    
    # Loan details
    st.markdown("### 💰 Loan Against Property")
    
    col1, col2 = st.columns(2)
    
    with col1:
        desired_loan_lakhs = st.number_input(
            "Desired Loan Amount (₹ Lakhs)",
            min_value=1.0,
            max_value=property_value_lakhs * 0.75,
            value=min(20.0, property_value_lakhs * 0.5),
            step=1.0,
            help="How much loan do you want?"
        )
    
    with col2:
        loan_tenure_years = st.selectbox(
            "Loan Tenure (Years)",
            [5, 10, 15, 20, 25, 30],
            index=3,
            help="Repayment period"
        )
    
    st.markdown("---")
    
    # Calculate loan eligibility
    property_value = property_value_lakhs * 100000
    desired_loan = desired_loan_lakhs * 100000
    
    # LTV (Loan to Value) ratio - banks typically offer 50-75%
    base_ltv = 0.75
    
    # Adjust LTV based on factors
    ltv_adjustments = []
    
    if property_age > 20:
        base_ltv -= 0.10
        ltv_adjustments.append("⚠️ Property age > 20 years: -10%")
    elif property_age > 10:
        base_ltv -= 0.05
        ltv_adjustments.append("⚠️ Property age > 10 years: -5%")
    
    if property_location in ["Rural Area", "Town"]:
        base_ltv -= 0.10
        ltv_adjustments.append("⚠️ Rural/Town location: -10%")
    elif property_location == "Tier 2 City":
        base_ltv -= 0.05
        ltv_adjustments.append("⚠️ Tier 2 city: -5%")
    
    if property_condition in ["Needs Renovation", "Average"]:
        base_ltv -= 0.05
        ltv_adjustments.append("⚠️ Property condition: -5%")
    
    if ownership_status != "Fully Owned":
        base_ltv -= 0.15
        ltv_adjustments.append("⚠️ Not fully owned: -15%")
    
    # Final LTV
    final_ltv = max(0.4, min(0.75, base_ltv))
    max_eligible_loan = property_value * final_ltv
    
    # Display results
    st.markdown("### 📊 Collateral Assessment Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <h4>🏠 Property Value</h4>
            <h1>₹{property_value_lakhs:.1f}L</h1>
            <p>Market Value</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="success-card">
            <h4>✅ Max Eligible Loan</h4>
            <h1>₹{max_eligible_loan/100000:.1f}L</h1>
            <p>{final_ltv*100:.0f}% LTV</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if desired_loan <= max_eligible_loan:
            st.markdown(f"""
            <div class="success-card">
                <h4>✅ Loan Status</h4>
                <h1>Approved</h1>
                <p>Within limit</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="danger-card">
                <h4>❌ Loan Status</h4>
                <h1>Exceeds Limit</h1>
                <p>Reduce amount</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Loan Details")
        
        if desired_loan <= max_eligible_loan:
            approved_loan = desired_loan
            status_color = "success"
            status_text = "✅ APPROVED"
        else:
            approved_loan = max_eligible_loan
            status_color = "danger"
            status_text = "⚠️ REDUCED"
        
        # Calculate EMI
        emi_calc = EMICalculator()
        interest_rate = 9.5  # Property loan rate
        tenure_months = loan_tenure_years * 12
        monthly_emi = emi_calc.calculate_emi(approved_loan, interest_rate, tenure_months)
        total_payment = monthly_emi * tenure_months
        total_interest = total_payment - approved_loan
        
        st.markdown(f"""
        <div class="{status_color}-card">
            <h4>{status_text}</h4>
            <p><strong>Approved Loan:</strong> ₹{approved_loan/100000:.1f}L</p>
            <p><strong>Interest Rate:</strong> {interest_rate}% p.a.</p>
            <p><strong>Monthly EMI:</strong> ₹{monthly_emi/1000:.1f}K</p>
            <p><strong>Total Interest:</strong> ₹{total_interest/100000:.2f}L</p>
            <p><strong>Total Repayment:</strong> ₹{total_payment/100000:.2f}L</p>
        </div>
        """, unsafe_allow_html=True)
        
        if ltv_adjustments:
            st.warning("**LTV Adjustments:**\n" + "\n".join(ltv_adjustments))
    
    with col2:
        st.markdown("#### ⚠️ Important Terms & Conditions")
        
        st.error(f"""
        **Loan Repayment Deadline:**
        - **Tenure:** {loan_tenure_years} years ({tenure_months} months)
        - **Final Payment:** {(datetime.now() + timedelta(days=loan_tenure_years*365)).strftime('%d %B %Y')}
        
        **Collateral Terms:**
        - Property will remain mortgaged to the bank
        - Original documents held by bank until full repayment
        - You can continue living/using the property
        - Property cannot be sold without bank permission
        
        **Default Consequences:**
        - Missing 3+ consecutive EMIs = Default
        - Bank can initiate legal proceedings
        - Property can be auctioned to recover dues
        - Credit score severely impacted
        - Legal costs added to outstanding amount
        """)
        
        st.info(f"""
        **Protection Advice:**
        - Always pay EMI before due date
        - Maintain 6 months EMI as emergency fund
        - Consider loan insurance
        - Keep property well-maintained
        - Update bank of address/contact changes
        """)
    
    st.markdown("---")
    
    # LTV breakdown chart
    st.markdown("### 📊 Loan-to-Value (LTV) Breakdown")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Pie chart showing LTV
        fig_ltv = go.Figure(data=[go.Pie(
            labels=['Approved Loan', 'Your Equity'],
            values=[approved_loan, property_value - approved_loan],
            hole=0.4,
            marker=dict(colors=['#f59e0b', '#10b981']),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>₹%{value:,.0f}<extra></extra>'
        )])
        
        fig_ltv.update_layout(
            title=f"Property Value: ₹{property_value_lakhs:.1f}L",
            height=350
        )
        
        st.plotly_chart(fig_ltv, use_container_width=True)
    
    with col2:
        st.markdown("#### 💡 Understanding LTV")
        
        st.markdown(f"""
        **LTV Ratio: {final_ltv*100:.0f}%**
        
        - Property Value: ₹{property_value_lakhs:.1f}L (100%)
        - Max Loan: ₹{max_eligible_loan/100000:.1f}L ({final_ltv*100:.0f}%)
        - Your Equity: ₹{(property_value - max_eligible_loan)/100000:.1f}L ({(1-final_ltv)*100:.0f}%)
        
        **Why not 100%?**
        Banks keep a safety margin to protect against:
        - Market value fluctuations
        - Property depreciation
        - Default risk
        - Legal/selling costs if foreclosure needed
        
        **Higher LTV possible if:**
        - ✅ New property (< 5 years)
        - ✅ Metro city location
        - ✅ Excellent condition
        - ✅ Clear title/fully owned
        - ✅ Higher credit score
        """)
    
    st.markdown("---")
    
    # Property value impact
    st.markdown("### 💰 How Property Value Affects Loan Amount")
    
    property_values = [20, 30, 40, 50, 75, 100, 150, 200]
    loan_comparison = []
    
    for pv in property_values:
        max_loan = pv * 100000 * final_ltv
        loan_comparison.append({
            'Property Value': f"₹{pv}L",
            f'Max Loan ({final_ltv*100:.0f}% LTV)': f"₹{max_loan/100000:.1f}L",
            'Your Equity': f"₹{(pv * 100000 - max_loan)/100000:.1f}L"
        })
    
    st.dataframe(
        pd.DataFrame(loan_comparison),
        use_container_width=True,
        hide_index=True
    )

def application_history_page():
    """Application history page"""
    st.markdown('<div class="main-header"><h1>🕐 Application History</h1><p>View Past Applications</p></div>', unsafe_allow_html=True)
    
    if not st.session_state.application_history:
        st.info("📋 No applications submitted yet. Start by submitting a new application!")
        return
    
    # Convert to DataFrame
    df_history = pd.DataFrame(st.session_state.application_history)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Applications", len(df_history))
    
    with col2:
        approval_rate = (df_history['decision'] == 'Approved').sum() / len(df_history) * 100
        st.metric("Approval Rate", f"{approval_rate:.1f}%")
    
    with col3:
        avg_loan = df_history['loan_amount'].mean()
        st.metric("Avg Loan Amount", f"₹{avg_loan:.0f}K")
    
    with col4:
        fraud_rate = df_history['fraud_detected'].sum() / len(df_history) * 100
        st.metric("Fraud Detection Rate", f"{fraud_rate:.1f}%")
    
    st.markdown("---")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        decision_filter = st.multiselect(
            "Filter by Decision",
            options=df_history['decision'].unique(),
            default=df_history['decision'].unique()
        )
    
    with col2:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=df_history['risk_level'].unique(),
            default=df_history['risk_level'].unique()
        )
    
    with col3:
        fraud_filter = st.selectbox(
            "Fraud Status",
            options=['All', 'Fraud Detected', 'No Fraud'],
            index=0
        )
    
    # Apply filters
    filtered_df = df_history[
        (df_history['decision'].isin(decision_filter)) &
        (df_history['risk_level'].isin(risk_filter))
    ]
    
    if fraud_filter == 'Fraud Detected':
        filtered_df = filtered_df[filtered_df['fraud_detected'] == True]
    elif fraud_filter == 'No Fraud':
        filtered_df = filtered_df[filtered_df['fraud_detected'] == False]
    
    # Display table
    st.markdown("### 📊 Application Records")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download History as CSV",
        data=csv,
        file_name=f"loan_history_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def analytics_page():
    """Analytics and insights page"""
    st.markdown('<div class="main-header"><h1>📈 Analytics Dashboard</h1><p>Insights & Trends</p></div>', unsafe_allow_html=True)
    
    if not st.session_state.application_history:
        st.info("📋 No data available yet. Submit applications to see analytics!")
        return
    
    df = pd.DataFrame(st.session_state.application_history)
    
    # Time series
    st.markdown("### 📅 Application Trends Over Time")
    
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_apps = df.groupby('date').size().reset_index(name='count')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_apps['date'],
        y=daily_apps['count'],
        mode='lines+markers',
        name='Applications',
        line=dict(color='#667eea', width=3),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title="Daily Application Volume",
        xaxis_title="Date",
        yaxis_title="Number of Applications",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Approval vs Rejection
        st.markdown("#### Decision Distribution")
        decision_counts = df['decision'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=decision_counts.index,
            values=decision_counts.values,
            hole=0.4,
            marker=dict(colors=['#10b981', '#ef4444'])
        )])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk distribution
        st.markdown("#### Risk Level Distribution")
        risk_counts = df['risk_level'].value_counts()
        
        fig = go.Figure(data=[go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            marker=dict(color=['#10b981', '#f59e0b', '#ef4444'])
        )])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown("---")
    st.markdown("### 🔗 Correlation Analysis")
    
    numeric_cols = ['applicant_income', 'loan_amount', 'fraud_score', 'approval_probability', 'risk_score']
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def about_page():
    """About page"""
    st.markdown('<div class="main-header"><h1>ℹ️ About Smart Loan Decision System</h1><p>Powered by AI & Machine Learning</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Overview
    
    The **Smart Loan Decision System** is an advanced AI-powered platform that revolutionizes the loan approval process.
    Using state-of-the-art machine learning algorithms, it provides:
    
    - **Automated fraud detection** using Isolation Forest
    - **Intelligent approval predictions** using ensemble methods
    - **Transparent AI explanations** using SHAP values
    - **Comprehensive risk assessment**
    - **Smart loan recommendations** based on affordability
    - **What-if scenario analysis** for better decision making
    
    ---
    
    ## 🔧 Technologies Used
    
    - **Frontend**: Streamlit, Plotly, Custom CSS
    - **Machine Learning**: Scikit-learn, XGBoost, LightGBM
    - **Explainable AI**: SHAP (SHapley Additive exPlanations)
    - **Data Processing**: Pandas, NumPy
    - **Authentication**: Bcrypt
    
    ---
    
    ## 📊 Model Architecture
    
    ### 1. Fraud Detection
    - **Algorithm**: Isolation Forest
    - **Purpose**: Detect anomalous applications
    - **Accuracy**: 85%+
    
    ### 2. Approval Prediction
    - **Algorithms**: Random Forest, Gradient Boosting, Logistic Regression
    - **Ensemble**: Voting classifier
    - **Accuracy**: 90%+
    
    ### 3. Risk Scoring
    - **Method**: Weighted multi-factor analysis
    - **Factors**: Credit history, DTI, income stability, employment, etc.
    
    ---
    
    ## 🚀 Features
    
    ### ✅ For Applicants
    - Easy-to-use application form
    - Instant decision feedback
    - Transparent explanations
    - Personalized recommendations
    - Financial planning tools
    
    ### ✅ For Loan Officers
    - Automated screening
    - Risk assessment dashboard
    - Fraud detection alerts
    - Application analytics
    - Historical tracking
    
    ---
    
    ## 📈 Benefits
    
    - **Faster Processing**: Instant decisions instead of days
    - **Higher Accuracy**: AI reduces human error and bias
    - **Transparency**: Clear explanations for every decision
    - **Fraud Prevention**: Advanced anomaly detection
    - **Better Planning**: What-if scenarios help applicants
    
    ---
    
    ## 📞 Support
    
    For questions or support, please contact:
    - Email: support@smartloan.com
    - Phone: +91-1234567890
    
    ---
    
    ## 📄 Version
    
    **Version 1.0.0** - February 2026
    
    ---
    
    *Built with ❤️ using AI and Machine Learning*
    """)

# Main execution
def main():
    """Main application entry point"""
    
    # Check if user is logged in
    if not st.session_state.logged_in:
        # Show signup or login page
        if 'page' not in st.session_state:
            st.session_state.page = 'login'
        
        if st.session_state.page == 'signup':
            signup_page()
        else:
            login_page()
    else:
        # Show main application
        main_app()

if __name__ == "__main__":
    main()