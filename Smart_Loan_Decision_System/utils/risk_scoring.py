"""
Risk Scoring Module
Calculates comprehensive risk scores for loan applications
"""

import numpy as np
import pandas as pd

class RiskScorer:
    def __init__(self):
        """Initialize Risk Scorer with weight configurations"""
        self.weights = {
            'credit_score': 0.35,
            'loan_to_income': 0.25,
            'dti_ratio': 0.20,
            'income_stability': 0.10,
            'employment': 0.05,
            'education': 0.03,
            'property': 0.02
        }
        
    def calculate_risk_score(self, application_data):
        """
        Calculate comprehensive risk score (0-100)
        Lower score = Lower risk (better)
        Higher score = Higher risk (worse)
        
        Returns:
            risk_score: Overall risk score
            risk_level: Low / Medium / High
            risk_breakdown: Individual component scores
        """
        scores = {}
        
        # 1. Credit Score Assessment (35%) - FICO Scale 300-850
        credit_score = application_data.get('Credit_Score', 650)
        if credit_score >= 800:
            scores['credit_score'] = 5  # Excellent
        elif credit_score >= 740:
            scores['credit_score'] = 15  # Very Good
        elif credit_score >= 670:
            scores['credit_score'] = 30  # Good
        elif credit_score >= 580:
            scores['credit_score'] = 55  # Fair
        else:
            scores['credit_score'] = 85  # Poor
        
        # 2. Loan to Income Ratio (25%)
        total_income = application_data.get('Total_Income', application_data.get('ApplicantIncome', 0))
        loan_amount = application_data.get('LoanAmount', 0) * 1000  # Convert thousands to actual
        annual_income = total_income * 12
        
        if annual_income > 0:
            loan_to_income = loan_amount / annual_income
        else:
            loan_to_income = 10
        
        if loan_to_income < 1.0:
            scores['loan_to_income'] = 10  # Excellent
        elif loan_to_income < 2.0:
            scores['loan_to_income'] = 25  # Good
        elif loan_to_income < 3.0:
            scores['loan_to_income'] = 45  # Moderate
        elif loan_to_income < 4.0:
            scores['loan_to_income'] = 65  # High
        else:
            scores['loan_to_income'] = 90  # Very High
        
        # 3. Debt-to-Income Ratio (20%)
        dti = application_data.get('DTI', 0)
        if dti < 0.30:
            scores['dti_ratio'] = 10
        elif dti < 0.40:
            scores['dti_ratio'] = 25
        elif dti < 0.50:
            scores['dti_ratio'] = 50
        else:
            scores['dti_ratio'] = 85
        
        # 4. Income Stability (10%)
        coapplicant_income = application_data.get('CoapplicantIncome', 0)
        applicant_income = application_data.get('ApplicantIncome', 0)
        
        if coapplicant_income > 0:
            # Dual income
            if total_income >= 80000:
                scores['income_stability'] = 10  # High dual income
            elif total_income >= 50000:
                scores['income_stability'] = 20  # Good dual income
            else:
                scores['income_stability'] = 35  # Modest dual income
        else:
            # Single income
            if applicant_income >= 80000:
                scores['income_stability'] = 25  # High single income
            elif applicant_income >= 50000:
                scores['income_stability'] = 40  # Good single income
            else:
                scores['income_stability'] = 60  # Low single income
        
        # 5. Employment Type (5%)
        self_employed = application_data.get('Self_Employed', 0)
        if self_employed == 0:  # Salaried
            scores['employment'] = 20
        else:  # Self-employed
            scores['employment'] = 50
        
        # 6. Education (3%)
        education = application_data.get('Education', 0)
        if education == 1:  # Graduate
            scores['education'] = 20
        else:
            scores['education'] = 40
        
        # 7. Property Area (2%)
        property_area = application_data.get('Property_Area', 0)
        if property_area == 2:  # Urban
            scores['property'] = 20
        elif property_area == 1:  # Semiurban
            scores['property'] = 30
        else:  # Rural
            scores['property'] = 40
        
        # Calculate weighted risk score
        risk_score = sum(scores[key] * self.weights[key] for key in scores)
        risk_score = round(risk_score, 2)
        
        # Determine risk level
        if risk_score < 25:
            risk_level = "Excellent (Very Low Risk)"
            risk_color = "green"
        elif risk_score < 40:
            risk_level = "Good (Low Risk)"
            risk_color = "green"
        elif risk_score < 55:
            risk_level = "Fair (Medium Risk)"
            risk_color = "orange"
        elif risk_score < 70:
            risk_level = "Poor (High Risk)"
            risk_color = "red"
        else:
            risk_level = "Very Poor (Very High Risk)"
            risk_color = "red"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'breakdown': scores
        }
    
    def get_risk_factors(self, application_data, risk_breakdown):
        """
        Get detailed risk factors explanation
        """
        factors = []
        
        # Credit Score Analysis
        credit_score = application_data.get('Credit_Score', 650)
        if credit_score >= 800:
            factors.append({
                'category': 'Credit Score',
                'status': 'Excellent',
                'description': f'Credit score of {credit_score} is excellent (top tier)',
                'severity': 'low'
            })
        elif credit_score >= 740:
            factors.append({
                'category': 'Credit Score',
                'status': 'Very Good',
                'description': f'Credit score of {credit_score} is very good',
                'severity': 'low'
            })
        elif credit_score >= 670:
            factors.append({
                'category': 'Credit Score',
                'status': 'Good',
                'description': f'Credit score of {credit_score} is good',
                'severity': 'medium'
            })
        elif credit_score >= 580:
            factors.append({
                'category': 'Credit Score',
                'status': 'Fair',
                'description': f'Credit score of {credit_score} is fair - room for improvement',
                'severity': 'medium'
            })
        else:
            factors.append({
                'category': 'Credit Score',
                'status': 'Poor',
                'description': f'Credit score of {credit_score} needs significant improvement',
                'severity': 'high'
            })
        
        # Loan to Income Analysis
        if risk_breakdown['loan_to_income'] <= 25:
            factors.append({
                'category': 'Loan Affordability',
                'status': 'Excellent',
                'description': 'Loan amount is well within your income capacity',
                'severity': 'low'
            })
        elif risk_breakdown['loan_to_income'] <= 45:
            factors.append({
                'category': 'Loan Affordability',
                'status': 'Moderate',
                'description': 'Loan amount is reasonable but significant',
                'severity': 'medium'
            })
        else:
            factors.append({
                'category': 'Loan Affordability',
                'status': 'High',
                'description': 'Loan amount exceeds recommended limits for your income',
                'severity': 'high'
            })
        
        # DTI Analysis
        if risk_breakdown['dti_ratio'] <= 25:
            factors.append({
                'category': 'Monthly Payment Burden',
                'status': 'Comfortable',
                'description': 'Monthly EMI is easily affordable',
                'severity': 'low'
            })
        elif risk_breakdown['dti_ratio'] <= 50:
            factors.append({
                'category': 'Monthly Payment Burden',
                'status': 'Moderate',
                'description': 'Monthly EMI will be noticeable but manageable',
                'severity': 'medium'
            })
        else:
            factors.append({
                'category': 'Monthly Payment Burden',
                'status': 'High',
                'description': 'Monthly EMI may strain your budget significantly',
                'severity': 'high'
            })
        
        # Income Stability
        if risk_breakdown['income_stability'] <= 25:
            factors.append({
                'category': 'Income Stability',
                'status': 'Strong',
                'description': 'Strong and stable income profile',
                'severity': 'low'
            })
        
        return factors
    
    def get_improvement_suggestions(self, risk_breakdown, application_data):
        """
        Provide suggestions to improve risk score
        """
        suggestions = []
        
        credit_score = application_data.get('Credit_Score', 650)
        
        if credit_score < 740:
            suggestions.append(f"🔸 Improve your credit score from {credit_score} to 740+ for better rates")
            suggestions.append("🔸 Pay bills on time, reduce credit utilization, avoid new credit inquiries")
        
        if risk_breakdown['loan_to_income'] > 45:
            suggestions.append("🔸 Consider reducing loan amount for better approval odds")
            suggestions.append("🔸 Or increase down payment to lower loan requirement")
        
        if risk_breakdown['dti_ratio'] > 50:
            suggestions.append("🔸 Choose longer tenure to reduce monthly EMI burden")
            suggestions.append("🔸 Or consider delaying purchase to improve financial position")
        
        if risk_breakdown['income_stability'] > 40:
            suggestions.append("🔸 Add co-applicant income to strengthen application")
            suggestions.append("🔸 Document all sources of income including bonuses")
        
        if not suggestions:
            suggestions.append("✅ Excellent application profile - minimal improvements needed")
        
        return suggestions
    
    def compare_with_benchmark(self, risk_score):
        """
        Compare applicant's risk score with benchmarks
        """
        benchmarks = {
            'Excellent': 20,
            'Good': 35,
            'Average': 50,
            'Below Average': 65,
            'Poor': 80
        }
        
        comparison = []
        for category, threshold in benchmarks.items():
            comparison.append({
                'category': category,
                'threshold': threshold,
                'status': 'current' if risk_score <= threshold + 5 else 'target'
            })
        
        return comparison