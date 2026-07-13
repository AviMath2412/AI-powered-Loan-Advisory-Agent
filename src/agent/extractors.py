import re
from typing import Optional

_LAKH = 100_000
_CRORE = 10_000_000

_UNIT_MULTIPLIERS = {
    "lakh": _LAKH, "lakhs": _LAKH, "lac": _LAKH, "lacs": _LAKH,
    "crore": _CRORE, "crores": _CRORE,
}

CALC_KEYWORDS = ("emi", "calculate", "installment", "monthly payment", "amortiz")


def parse_principal(text: str) -> Optional[float]:
    text_l = text.lower()

    # "20 lakh", "1.5 crore"
    m = re.search(r'(\d+(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs|crore|crores)\b', text_l)
    if m:
        return float(m.group(1)) * _UNIT_MULTIPLIERS[m.group(2)]

    # "₹5,00,000", "Rs. 500000", "500000"
    m = re.search(r'(?:₹|rs\.?|inr)?\s*([\d,]{4,})(?!\s*%)', text_l)
    if m:
        num = m.group(1).replace(",", "")
        if num.isdigit():
            return float(num)

    return None


def parse_rate(text: str) -> Optional[float]:
    m = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
    if m:
        return float(m.group(1))
    m = re.search(r'(\d+(?:\.\d+)?)\s*percent', text.lower())
    if m:
        return float(m.group(1))
    return None


def parse_tenure_months(text: str) -> Optional[int]:
    text_l = text.lower()
    m = re.search(r'(\d+(?:\.\d+)?)\s*(years|year|yrs|yr)\b', text_l)
    if m:
        return int(round(float(m.group(1)) * 12))
    m = re.search(r'(\d+)\s*(months|month|mos|mo)\b', text_l)
    if m:
        return int(m.group(1))
    return None


def extract_calc_params(text: str) -> Optional[dict]:
    """Returns a complete {principal, rate_pa, tenure_months} dict only if all three
    are found, otherwise None — a partial dict isn't useful to the Calculator."""
    principal = parse_principal(text)
    rate = parse_rate(text)
    tenure = parse_tenure_months(text)
    if principal and rate and tenure:
        return {"principal": principal, "rate_pa": rate, "tenure_months": tenure}
    return None


def looks_like_calc_request(text: str) -> bool:
    text_l = text.lower()
    return any(keyword in text_l for keyword in CALC_KEYWORDS)
