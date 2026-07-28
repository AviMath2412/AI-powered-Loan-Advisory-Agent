import time
import random
import logging
from typing import Any, Optional, Dict, List, Callable
from src.observability import metrics_exporter
from src.config import MAX_RETRIES, LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


class CircuitBreakerOpenException(Exception):
    """Raised when an execution is attempted while the circuit breaker is OPEN."""
    pass


class CircuitBreaker:
    """
    Implements the Circuit Breaker pattern.
    - CLOSED: Normal operation. Requests pass through.
    - OPEN: Service has failed repeatedly. Requests fail fast or use fallbacks.
    - HALF-OPEN: Recovery trial after cooldown. Single success resets to CLOSED; failure re-opens.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        name: str = "LLM_Service"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED | OPEN | HALF-OPEN

    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF-OPEN"
                logger.info(f"CircuitBreaker [{self.name}] transition to HALF-OPEN (testing service recovery).")
                return True
            return False
        elif self.state == "HALF-OPEN":
            return True
        return True

    def record_success(self):
        if self.state != "CLOSED":
            logger.info(f"CircuitBreaker [{self.name}] recovered successfully! State set to CLOSED.")
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"CircuitBreaker [{self.name}] tripped to OPEN after {self.failure_count} consecutive failures. "
                f"Cooldown period: {self.recovery_timeout}s."
            )

    def reset(self):
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"


# Default global circuit breaker for LLM services
default_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0, name="Default_LLM")


def invoke_llm_with_resilience(
    llm: Any,
    messages: Any,
    circuit_breaker: Optional[CircuitBreaker] = None,
    max_retries: int = MAX_RETRIES,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    fallback_response: Optional[str] = None
) -> str:
    """
    Executes LLM invocation with:
    1. Circuit Breaker protection
    2. Exponential Backoff retries with jitter
    3. Graceful degradation via fallback response on failure
    """
    cb = circuit_breaker or default_circuit_breaker

    if not cb.can_execute():
        logger.warning(f"CircuitBreaker [{cb.name}] is OPEN. Bypassing LLM call.")
        if fallback_response is not None:
            return fallback_response
        raise CircuitBreakerOpenException(f"CircuitBreaker [{cb.name}] is currently OPEN.")

    delay = initial_delay
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            start_time = time.time()
            if hasattr(llm, "with_config"):
                response = llm.with_config({"timeout": LLM_TIMEOUT_SECONDS}).invoke(messages)
            else:
                response = llm.invoke(messages)
            
            elapsed = time.time() - start_time
            logger.info(f"LLM call succeeded in {elapsed:.2f}s")
            
            cb.record_success()
            
            # Record token usage if available in the response metadata
            if hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
                usage = response.response_metadata["token_usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                model_name = getattr(llm, "model", getattr(llm, "model_id", "unknown_model"))
                metrics_exporter.record_token_usage(model_name, prompt_tokens, completion_tokens)
                
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)
        except Exception as e:
            last_exception = e
            cb.record_failure()
            logger.warning(f"LLM API call attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                sleep_time = delay * (1 + random.uniform(0, 0.2))
                time.sleep(sleep_time)
                delay *= backoff_factor

    # All retries failed
    if fallback_response is not None:
        logger.error(f"All {max_retries} LLM call retries failed. Using fallback response.")
        return fallback_response

    raise last_exception
