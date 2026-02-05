"""
EMI Calculator and Loan Recommendation Module
Calculates EMI and recommends safe loan amounts based on income
"""

import numpy as np
import pandas as pd

class EMICalculator:
    def __init__(self):
        """Initialize EMI Calculator"""
        self.max_emi_ratio = 0.4  # Maximum 40% of income for EMI
        self.safe_emi_ratio = 0.35  # Safe 35% of income for EMI
        
    def calculate_emi(self, principal, annual_rate, tenure_months):
        """
        Calculate EMI using formula: EMI = P × r × (1 + r)^n / ((1 + r)^n - 1)
        
        Args:
            principal: Loan amount
            annual_rate: Annual interest rate (percentage)
            tenure_months: Loan tenure in months
        
        Returns:
            emi: Monthly EMI amount
        """
        # Convert annual rate to monthly rate
        monthly_rate = annual_rate / (12 * 100)
        
        if monthly_rate == 0:
            return principal / tenure_months
        
        # EMI calculation
        emi = principal * monthly_rate * ((1 + monthly_rate) ** tenure_months) / \
              (((1 + monthly_rate) ** tenure_months) - 1)
        
        return round(emi, 2)
    
    def recommend_interest_rate(self, credit_score, employment_type, income_level):
        """
        Recommend interest rate based on applicant profile using Credit Score (300-850)
        
        Returns:
            interest_rate: Recommended annual interest rate
            rate_category: Risk category
        """
        # Base rate based on credit score (FICO-style)
        if credit_score >= 800:
            base_rate = 7.5  # Excellent credit
            rate_category = "Excellent Credit - Premium Rate"
        elif credit_score >= 740:
            base_rate = 8.5  # Very Good credit
            rate_category = "Very Good Credit - Preferred Rate"
        elif credit_score >= 670:
            base_rate = 9.5  # Good credit
            rate_category = "Good Credit - Standard Rate"
        elif credit_score >= 580:
            base_rate = 10.5  # Fair credit
            rate_category = "Fair Credit - Higher Rate"
        else:
            base_rate = 12.5  # Poor credit
            rate_category = "Poor Credit - Subprime Rate"
        
        # Adjust based on employment type
        if employment_type == 1:  # Self-employed
            base_rate += 0.5
        
        # Adjust based on income level
        if income_level < 30000:
            base_rate += 1.0
        elif income_level < 50000:
            base_rate += 0.5
        
        return round(base_rate, 2), rate_category
    
    def calculate_max_loan_amount(self, monthly_income, annual_rate, tenure_months, existing_emi=0):
        """
        Calculate maximum affordable loan amount
        
        Args:
            monthly_income: Applicant's monthly income
            annual_rate: Interest rate
            tenure_months: Loan tenure
            existing_emi: Any existing EMI obligations
        
        Returns:
            max_loan: Maximum affordable loan amount
        """
        # Calculate maximum EMI applicant can afford
        max_affordable_emi = (monthly_income * self.max_emi_ratio) - existing_emi
        
        if max_affordable_emi <= 0:
            return 0
        
        # Calculate principal from EMI
        monthly_rate = annual_rate / (12 * 100)
        
        if monthly_rate == 0:
            max_loan = max_affordable_emi * tenure_months
        else:
            max_loan = max_affordable_emi * (((1 + monthly_rate) ** tenure_months) - 1) / \
                      (monthly_rate * ((1 + monthly_rate) ** tenure_months))
        
        return round(max_loan, 2)
    
    def recommend_safe_loan_amount(self, monthly_income, annual_rate, tenure_months, 
                                   requested_amount, existing_emi=0):
        """
        Recommend a safe loan amount based on income and other factors
        
        Returns:
            recommendation: Dict with loan details
        """
        # Calculate maximum possible loan
        max_loan = self.calculate_max_loan_amount(
            monthly_income, annual_rate, tenure_months, existing_emi
        )
        
        # Calculate safe loan (35% of income instead of 40%)
        safe_affordable_emi = (monthly_income * self.safe_emi_ratio) - existing_emi
        monthly_rate = annual_rate / (12 * 100)
        
        if monthly_rate == 0:
            safe_loan = safe_affordable_emi * tenure_months
        else:
            safe_loan = safe_affordable_emi * (((1 + monthly_rate) ** tenure_months) - 1) / \
                       (monthly_rate * ((1 + monthly_rate) ** tenure_months))
        
        safe_loan = round(safe_loan, 2)
        
        # Determine recommendation
        if requested_amount <= safe_loan:
            recommended_amount = requested_amount
            emi = self.calculate_emi(requested_amount, annual_rate, tenure_months)
            recommendation_type = "Approved as Requested"
            message = f"The requested amount of ₹{requested_amount:,.2f} is within safe limits."
        elif requested_amount <= max_loan:
            recommended_amount = requested_amount
            emi = self.calculate_emi(requested_amount, annual_rate, tenure_months)
            recommendation_type = "Approved with Caution"
            message = f"The requested amount is approved but near maximum capacity. Consider ₹{safe_loan:,.2f} for better financial health."
        else:
            recommended_amount = safe_loan
            emi = self.calculate_emi(safe_loan, annual_rate, tenure_months)
            recommendation_type = "Reduced Amount Recommended"
            message = f"Requested amount exceeds safe limit. Recommended amount: ₹{safe_loan:,.2f}"
        
        # Calculate total payment and interest
        total_payment = emi * tenure_months
        total_interest = total_payment - recommended_amount
        
        return {
            'recommended_amount': recommended_amount,
            'monthly_emi': emi,
            'total_payment': round(total_payment, 2),
            'total_interest': round(total_interest, 2),
            'interest_rate': annual_rate,
            'tenure_months': tenure_months,
            'tenure_years': tenure_months / 12,
            'max_affordable_loan': max_loan,
            'safe_loan_amount': safe_loan,
            'recommendation_type': recommendation_type,
            'message': message,
            'emi_to_income_ratio': round((emi / monthly_income) * 100, 2)
        }
    
    def create_amortization_schedule(self, principal, annual_rate, tenure_months, num_rows=12):
        """
        Create amortization schedule for first year
        
        Returns:
            DataFrame with payment schedule
        """
        emi = self.calculate_emi(principal, annual_rate, tenure_months)
        monthly_rate = annual_rate / (12 * 100)
        
        schedule = []
        balance = principal
        
        for month in range(1, min(num_rows + 1, tenure_months + 1)):
            interest = balance * monthly_rate
            principal_payment = emi - interest
            balance -= principal_payment
            
            schedule.append({
                'Month': month,
                'EMI': round(emi, 2),
                'Principal': round(principal_payment, 2),
                'Interest': round(interest, 2),
                'Balance': round(max(balance, 0), 2)
            })
        
        return pd.DataFrame(schedule)
    
    def what_if_analysis(self, base_income, base_loan, annual_rate, tenure_months):
        """
        Perform what-if analysis with different scenarios
        
        Returns:
            List of scenarios with EMI and affordability
        """
        scenarios = []
        
        # Scenario 1: Current request
        emi_current = self.calculate_emi(base_loan, annual_rate, tenure_months)
        scenarios.append({
            'scenario': 'Current Request',
            'loan_amount': base_loan,
            'monthly_emi': emi_current,
            'emi_percentage': round((emi_current / base_income) * 100, 2),
            'affordability': 'Safe' if (emi_current / base_income) <= 0.35 else 'Risky' if (emi_current / base_income) <= 0.45 else 'Very Risky'
        })
        
        # Scenario 2: 20% less loan
        loan_reduced = base_loan * 0.8
        emi_reduced = self.calculate_emi(loan_reduced, annual_rate, tenure_months)
        scenarios.append({
            'scenario': '20% Reduced Loan',
            'loan_amount': round(loan_reduced, 2),
            'monthly_emi': emi_reduced,
            'emi_percentage': round((emi_reduced / base_income) * 100, 2),
            'affordability': 'Safe' if (emi_reduced / base_income) <= 0.35 else 'Risky' if (emi_reduced / base_income) <= 0.45 else 'Very Risky'
        })
        
        # Scenario 3: Longer tenure
        tenure_longer = tenure_months + 60  # 5 more years
        emi_longer = self.calculate_emi(base_loan, annual_rate, tenure_longer)
        scenarios.append({
            'scenario': f'{tenure_longer // 12} Year Tenure',
            'loan_amount': base_loan,
            'monthly_emi': emi_longer,
            'emi_percentage': round((emi_longer / base_income) * 100, 2),
            'affordability': 'Safe' if (emi_longer / base_income) <= 0.35 else 'Risky' if (emi_longer / base_income) <= 0.45 else 'Very Risky'
        })
        
        # Scenario 4: With 20% higher income
        income_increased = base_income * 1.2
        scenarios.append({
            'scenario': '20% Income Increase',
            'loan_amount': base_loan,
            'monthly_emi': emi_current,
            'emi_percentage': round((emi_current / income_increased) * 100, 2),
            'affordability': 'Safe' if (emi_current / income_increased) <= 0.35 else 'Risky' if (emi_current / income_increased) <= 0.45 else 'Very Risky'
        })
        
        return pd.DataFrame(scenarios)