import pytest
from unittest.mock import patch

@pytest.fixture(autouse=True)
def disable_circuit_breaker():
    """
    Mock the circuit breaker to prevent it from opening during tests.
    This ensures that the LLM is actually called during tests instead of 
    being bypassed by the open circuit breaker when tests run rapidly.
    """
    with patch("src.agent.resilience.CircuitBreaker.can_execute", return_value=True):
        yield
