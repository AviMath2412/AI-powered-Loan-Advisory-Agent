import hashlib

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


def compute_amortization_schedule(principal: float, rate_pa: float, tenure_months: int) -> list[dict]:
    """
    Pure calculation, no LLM involved. Used by calculate_emi (for the LLM-facing markdown table)
    and directly by the Streamlit UI (to build the dataframe/chart), so both always agree.

    Returns a list of yearly rows: {"year", "principal_paid", "interest_paid", "ending_balance"}.
    """
    if principal <= 0 or rate_pa < 0 or tenure_months <= 0:
        return []

    if rate_pa == 0:
        r = 0.0
        emi = principal / tenure_months
    else:
        r = rate_pa / (12 * 100)
        emi = principal * r * ((1 + r) ** tenure_months) / (((1 + r) ** tenure_months) - 1)

    schedule = []
    balance = principal
    yearly_principal = 0.0
    yearly_interest = 0.0

    for month in range(1, tenure_months + 1):
        interest_payment = balance * r
        principal_payment = emi - interest_payment

        yearly_interest += interest_payment
        yearly_principal += principal_payment
        balance -= principal_payment

        if month % 12 == 0 or month == tenure_months:
            year = month // 12 if month % 12 == 0 else (month // 12) + 1
            schedule.append({
                "year": year,
                "principal_paid": round(yearly_principal, 2),
                "interest_paid": round(yearly_interest, 2),
                "ending_balance": round(max(0.0, balance), 2),
            })
            yearly_principal = 0.0
            yearly_interest = 0.0

    return schedule


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
        if principal <= 0:
            return "Error calculating EMI: Loan principal must be greater than zero."
        if rate_pa < 0:
            return "Error calculating EMI: Interest rate cannot be negative."
        if tenure_months <= 0:
            return "Error calculating EMI: Loan tenure must be greater than zero."

        schedule = compute_amortization_schedule(principal, rate_pa, tenure_months)
        if not schedule:
            return "Error calculating EMI: tenure_months must be greater than 0."

        if rate_pa == 0:
            emi = principal / tenure_months
        else:
            r = rate_pa / (12 * 100)
            emi = principal * r * ((1 + r) ** tenure_months) / (((1 + r) ** tenure_months) - 1)

        total_amount = emi * tenure_months
        total_interest = total_amount - principal

        table = "\n\n### Yearly Amortization Schedule\n"
        table += "| Year | Principal Paid | Interest Paid | Ending Balance |\n"
        table += "|------|----------------|---------------|----------------|\n"
        for row in schedule:
            table += (f"| Year {row['year']} | ₹ {row['principal_paid']:,.2f} "
                       f"| ₹ {row['interest_paid']:,.2f} | ₹ {row['ending_balance']:,.2f} |\n")

        return (f"**Calculation Successful:**\n"
                f"- **Monthly EMI:** ₹ {emi:,.2f}\n"
                f"- **Total Interest Payable:** ₹ {total_interest:,.2f}\n"
                f"- **Total Amount to be Paid:** ₹ {total_amount:,.2f}\n"
                f"{table}")

    except Exception as e:
        return f"Error calculating EMI: {str(e)}"


@tool
def check_credit_score(applicant_id: str) -> str:
    """
    Mock Credit Bureau lookup. Returns a simulated credit score and risk band.
    This is a stand-in for a real Credit Bureau API — the return contract (score + band)
    is designed so swapping this body for a real HTTP call requires no changes upstream.

    Args:
        applicant_id: Any user-supplied identifier (e.g., a masked account reference).
    """
    seed = int(hashlib.sha256(applicant_id.encode()).hexdigest(), 16) % 350
    score = 550 + seed

    if score >= 750:
        band = "Excellent"
    elif score >= 700:
        band = "Good"
    elif score >= 650:
        band = "Fair"
    else:
        band = "Needs Improvement"

    return (f"Simulated Credit Score for `{applicant_id}`: **{score}** ({band}). "
            f"[MOCK DATA — replace check_credit_score's body with a real Credit Bureau API call "
            f"for production use.]")


# A list of all tools our agent is allowed to use.
AGENT_TOOLS = [search_loan_policies, calculate_emi, check_credit_score]