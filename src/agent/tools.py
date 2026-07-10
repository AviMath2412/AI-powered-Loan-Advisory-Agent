from langchain_core.tools import tool
from src.rag.retriever import retrieve_loan_context

@tool
def search_loan_policies(query: str) -> str:
    """
    Searches the internal bank database for loan rules, interest rates, eligibility criteria, and fees.
    Use this tool FIRST whenever the user asks a factual question about loans.
    
    Args:
        query: The search query to look up in the database.
    """
    return retrieve_loan_context(query)

@tool
def calculate_emi(principal: float, rate_pa: float, tenure_months: int) -> str:
    """
    Calculates the Equated Monthly Installment (EMI) for a loan based on standard banking formulas.
    Use this tool ONLY when the user asks to calculate an EMI or monthly payment.
    
    Args:
        principal: The total loan amount requested (e.g., 500000).
        rate_pa: The annual interest rate in percentage (e.g., 10.5).
        tenure_months: The duration of the loan in months (e.g., 60).
    """
    try:
        if rate_pa == 0:
            emi = principal / tenure_months
        else:
            r = rate_pa / (12 * 100)  # Convert annual rate to monthly decimal
            emi = principal * r * ((1 + r)**tenure_months) / (((1 + r)**tenure_months) - 1)
        
        total_amount = emi * tenure_months
        total_interest = total_amount - principal
        
        # --- UPGRADE: Generate Yearly Amortization Schedule ---
        schedule = "\n\n### 📊 Yearly Amortization Schedule\n"
        schedule += "| Year | Principal Paid | Interest Paid | Ending Balance |\n"
        schedule += "|------|----------------|---------------|----------------|\n"
        
        balance = principal
        yearly_principal = 0
        yearly_interest = 0
        
        for month in range(1, tenure_months + 1):
            interest_payment = balance * r
            principal_payment = emi - interest_payment
            
            yearly_interest += interest_payment
            yearly_principal += principal_payment
            balance -= principal_payment
            
            # Append to schedule at the end of each year or the final month
            if month % 12 == 0 or month == tenure_months:
                year = month // 12 if month % 12 == 0 else (month // 12) + 1
                # Format numbers with commas and 2 decimal places for a professional look
                schedule += f"| Year {year} | ₹ {yearly_principal:,.2f} | ₹ {yearly_interest:,.2f} | ₹ {max(0, balance):,.2f} |\n"
                
                # Reset yearly counters for the next loop
                yearly_principal = 0
                yearly_interest = 0
        
        # Return the clean, calculated data back to the LLM context
        return (f"**Calculation Successful:**\n"
                f"- **Monthly EMI:** ₹ {emi:,.2f}\n"
                f"- **Total Interest Payable:** ₹ {total_interest:,.2f}\n"
                f"- **Total Amount to be Paid:** ₹ {total_amount:,.2f}\n"
                f"{schedule}")
                
    except Exception as e:
        return f"Error calculating EMI: {str(e)}"

@tool
def check_credit_score(applicant_id: str) -> str:
    """
    Checks the credit score (CIBIL or equivalent) for the applicant.
    
    Args:
        applicant_id: The identifier of the applicant (e.g. '12345' or 'applicant_name').
    """
    if not applicant_id:
        return "Error: Applicant ID is required to check credit score."
    
    # Stable hash logic to generate a score between 300 and 850
    import hashlib
    h = hashlib.md5(applicant_id.lower().strip().encode()).hexdigest()
    score = 300 + (int(h, 16) % 551)
    
    if score >= 750:
        status = "Excellent"
    elif score >= 700:
        status = "Good"
    elif score >= 650:
        status = "Average"
    else:
        status = "Poor"
        
    return (f"Credit Score Check Completed for Applicant: {applicant_id}\n"
            f"- **Score:** {score}\n"
            f"- **Status:** {status}\n"
            f"- **Recommendation:** "
            f"{'Highly recommended for loan approval' if score >= 700 else 'Approval subject to higher interest rates' if score >= 650 else 'High risk - approval not recommended'}")

def compute_amortization_schedule(principal: float, rate_pa: float, tenure_months: int) -> list[dict]:
    """
    Computes a yearly amortization schedule returning list of year stats.
    Used for graphing in the UI dashboard.
    """
    try:
        if rate_pa == 0:
            emi = principal / tenure_months
            r = 0
        else:
            r = rate_pa / (12 * 100)
            emi = principal * r * ((1 + r)**tenure_months) / (((1 + r)**tenure_months) - 1)
            
        balance = principal
        schedule = []
        
        yearly_principal = 0
        yearly_interest = 0
        
        for month in range(1, tenure_months + 1):
            if r == 0:
                interest_payment = 0
                principal_payment = emi
            else:
                interest_payment = balance * r
                principal_payment = emi - interest_payment
                
            yearly_interest += interest_payment
            yearly_principal += principal_payment
            balance -= principal_payment
            
            if month % 12 == 0 or month == tenure_months:
                year = month // 12 if month % 12 == 0 else (month // 12) + 1
                schedule.append({
                    "year": int(year),
                    "principal_paid": round(yearly_principal, 2),
                    "interest_paid": round(yearly_interest, 2),
                    "ending_balance": round(max(0.0, balance), 2)
                })
                yearly_principal = 0
                yearly_interest = 0
        return schedule
    except Exception:
        return []

# A list of all tools our agent is allowed to use.
AGENT_TOOLS = [search_loan_policies, calculate_emi, check_credit_score]