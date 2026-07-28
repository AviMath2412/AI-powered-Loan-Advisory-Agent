import math
from typing import Dict, Any, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta

def normalize_currency(amount: float, exchange_rate: float) -> float:
    """Normalizes currency based on a provided exchange rate."""
    return amount * exchange_rate

def calculate_percentage(part: float, whole: float) -> float:
    """Returns the percentage that 'part' is of 'whole'."""
    if whole == 0:
        return 0.0
    return (part / whole) * 100.0

def validate_ltv(loan_amount: float, property_value: float, max_ltv_percentage: float) -> Tuple[bool, float]:
    """Validates Loan-to-Value (LTV). Returns (is_valid, actual_ltv_percentage)."""
    if property_value <= 0:
        return False, 0.0
    ltv = (loan_amount / property_value) * 100.0
    return ltv <= max_ltv_percentage, ltv

def compare_apr(apr1: float, apr2: float) -> Dict[str, Any]:
    """Compares two APRs and returns the difference and the better one."""
    diff = abs(apr1 - apr2)
    better = "apr1" if apr1 < apr2 else "apr2" if apr2 < apr1 else "equal"
    return {"difference": diff, "lower_apr": better}

def timeline_arithmetic(start_date_str: str, months_to_add: int, fmt: str = "%Y-%m-%d") -> str:
    """Adds a certain number of months to a date string."""
    try:
        dt = datetime.strptime(start_date_str, fmt)
        new_dt = dt + relativedelta(months=months_to_add)
        return new_dt.strftime(fmt)
    except ValueError:
        return "Invalid date format"

def calculate_simple_interest(principal: float, rate_pa: float, time_years: float) -> float:
    """Calculates simple interest."""
    return principal * (rate_pa / 100.0) * time_years

def calculate_compound_interest(principal: float, rate_pa: float, time_years: float, compounds_per_year: int = 12) -> float:
    """Calculates compound interest."""
    r = rate_pa / 100.0
    amount = principal * math.pow((1 + r / compounds_per_year), compounds_per_year * time_years)
    return amount - principal

def calculate_cash_flow(inflows: list[float], outflows: list[float]) -> float:
    """Calculates net cash flow from inflows and outflows."""
    return sum(inflows) - sum(outflows)

def calculate_probability(favorable_outcomes: int, total_outcomes: int) -> float:
    """Calculates simple probability."""
    if total_outcomes <= 0:
        return 0.0
    return favorable_outcomes / total_outcomes
