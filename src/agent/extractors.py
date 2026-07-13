import re
from typing import Optional

# Indian currency multipliers
_LAKH = 100_000
_CRORE = 10_000_000

_UNIT_MULTIPLIERS = {
    "lakh": _LAKH, "lakhs": _LAKH, "lac": _LAKH, "lacs": _LAKH,
    "crore": _CRORE, "crores": _CRORE,
}

CALC_KEYWORDS = ("emi", "calculate", "installment", "monthly payment", "amortiz")


def parse_principal(text: str) -> Optional[float]:
    """Extracts the loan principal amount. Does not assume Lakh or Crore unless specified by the user."""
    text_l = text.lower()

    # 1. Look for explicit lakh/crore suffix (e.g., "8.09 Lakh", "20 Lakh", "1.5 Crore")
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs|crore|crores)\b', text_l)
    if m:
        return float(m.group(1)) * _UNIT_MULTIPLIERS[m.group(2)]

    # 2. Look for large numbers with currency symbols (e.g., ₹809000, Rs. 500000)
    m = re.search(r'(?:₹|rs\.?|inr)\s*(-?\d+(?:\.\d+)?)(?!\s*%)', text_l)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # 3. Look for bare numbers in context (excluding rate/tenure)
    matches = re.finditer(r'\b(\d+(?:\.\d+)?)\b', text_l)
    for match in matches:
        val_str = match.group(1)
        try:
            val = float(val_str)
        except ValueError:
            continue
            
        start_idx = match.start()
        end_idx = match.end()
        
        # Check context after and before the number to exclude interest rate or tenure
        post_context = text_l[end_idx:end_idx+20]
        pre_context = text_l[max(0, start_idx-20):start_idx]
        
        if any(w in post_context or w in pre_context for w in ("%", "percent", "year", "month", "yr", "mo", "age")):
            continue
            
        # Determine if negative
        is_negative = False
        if start_idx > 0 and text_l[start_idx-1] == '-':
            is_negative = True
        elif start_idx > 1 and text_l[start_idx-2:start_idx] == '- ':
            is_negative = True
            
        if is_negative:
            val = -val

        return val

    return None


def parse_rate(text: str) -> Optional[float]:
    """Extracts annual interest rate in percentage."""
    text_l = text.lower()
    
    # "8.5%" or "-8.5%"
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*%', text_l)
    if m:
        return float(m.group(1))
        
    # "8.5 percent" or "8.5 interest" or "-8.5 p.a."
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:percent|interest|p\.a|per annum)\b', text_l)
    if m:
        return float(m.group(1))
        
    # Bare rate in context
    matches = re.finditer(r'\b(\d+(?:\.\d+)?)\b', text_l)
    for match in matches:
        try:
            val = float(match.group(1))
        except ValueError:
            continue
            
        start_idx = match.start()
        end_idx = match.end()
        post_context = text_l[end_idx:end_idx+20]
        pre_context = text_l[max(0, start_idx-20):start_idx]
        
        # Check if preceded by negative sign
        is_negative = False
        if start_idx > 0 and text_l[start_idx-1] == '-':
            is_negative = True
        elif start_idx > 1 and text_l[start_idx-2:start_idx] == '- ':
            is_negative = True
            
        if is_negative:
            val = -val
            
        # Rate is typically between -30 and 30, and accompanied by rate indicators
        if -30.0 <= val <= 30.0:
            if any(w in post_context or w in pre_context for w in ("rate", "interest", "pa", "annum", "p.a")):
                if not any(w in post_context or w in pre_context for w in ("year", "month", "yr", "mo", "age")):
                    return val
                    
    return None


def parse_tenure_months(text: str) -> Optional[int]:
    """Extracts tenure and converts it to months."""
    text_l = text.lower()
    
    # "5 years" or "-5.5 years"
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:years|year|yrs|yr)\b', text_l)
    if m:
        return int(round(float(m.group(1)) * 12))
        
    # "60 months"
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:months|month|mos|mo)\b', text_l)
    if m:
        return int(round(float(m.group(1))))
        
    # Bare tenure in context (e.g. "for 5", "for 60")
    matches = re.finditer(r'\b(\d+(?:\.\d+)?)\b', text_l)
    for match in matches:
        try:
            val = float(match.group(1))
        except ValueError:
            continue
            
        start_idx = match.start()
        end_idx = match.end()
        post_context = text_l[end_idx:end_idx+20]
        pre_context = text_l[max(0, start_idx-20):start_idx]
        
        # Check if negative
        is_negative = False
        if start_idx > 0 and text_l[start_idx-1] == '-':
            is_negative = True
        elif start_idx > 1 and text_l[start_idx-2:start_idx] == '- ':
            is_negative = True
            
        if is_negative:
            val = -val
            
        if any(w in post_context or w in pre_context for w in ("tenure", "period", "duration", "for")):
            if not any(w in post_context or w in pre_context for w in ("%", "percent", "rate", "interest", "lakh", "crore", "₹", "rs")):
                if abs(val) <= 40:  # Assume years if small
                    return int(round(val * 12))
                else:  # Assume months
                    return int(round(val))
                    
    return None


def extract_calc_params(text: str) -> Optional[dict]:
    """Returns a complete {principal, rate_pa, tenure_months} dict only if all three
    are found, otherwise None."""
    principal = parse_principal(text)
    rate = parse_rate(text)
    tenure = parse_tenure_months(text)
    if principal is not None and rate is not None and tenure is not None:
        return {"principal": principal, "rate_pa": rate, "tenure_months": tenure}
    return None


def looks_like_calc_request(text: str) -> bool:
    text_l = text.lower()
    return any(keyword in text_l for keyword in CALC_KEYWORDS)
