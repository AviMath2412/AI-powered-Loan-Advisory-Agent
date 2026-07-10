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
        schedule = "\n\n**Yearly Amortization Schedule:**\n"
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
                schedule += f"| Year {year} | Rs. {yearly_principal:,.2f} | Rs. {yearly_interest:,.2f} | Rs. {max(0, balance):,.2f} |\n"
                # Reset yearly counters
                yearly_principal = 0
                yearly_interest = 0
        
        # We return the raw calculation cleanly back to the LLM
        return (f"Calculation Successful.\n"
                f"Monthly EMI: Rs. {emi:,.2f}\n"
                f"Total Interest Payable: Rs. {total_interest:,.2f}\n"
                f"Total Amount to be Paid: Rs. {total_amount:,.2f}"
                f"{schedule}")
                
    except Exception as e:
        return f"Error calculating EMI: {str(e)}"

# A list of all tools our agent is allowed to use.
# We will bind this list to Qwen2.5-Coder in the next phase.
AGENT_TOOLS = [search_loan_policies, calculate_emi]