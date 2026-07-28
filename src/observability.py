from IPython.core import logger
import os
import sys
import json
import time
import uuid
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from functools import wraps


class StructuredJSONFormatter(logging.Formatter):
    """
    Custom JSON Formatter producing structured JSON logs for audit, compliance, and distributed tracing.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_entry.update(record.extra_fields)

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def get_logger(name: str = "LoanAdvisoryAgent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJSONFormatter())
        logger.addHandler(handler)
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))
    return logger


agent_logger = get_logger("AgentLogger")


@dataclass
class TraceSpan:
    trace_id: str
    span_id: str
    name: str
    start_time: float
    parent_span_id: Optional[str] = None
    end_time: Optional[float] = None
    latency_ms: Optional[float] = None
    status: str = "IN_PROGRESS"
    attributes: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class Tracer:
    """
    Lightweight distributed tracer.
    Tracks execution spans, node latencies, and parent-child trace relationships.
    """

    def __init__(self):
        self.spans: List[TraceSpan] = []
        self._current_trace_id: Optional[str] = None

    def start_trace(self, trace_id: Optional[str] = None) -> str:
        self._current_trace_id = trace_id or str(uuid.uuid4())
        return self._current_trace_id

    def start_span(
        self, name: str, parent_span_id: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None
    ) -> TraceSpan:
        if not self._current_trace_id:
            self.start_trace()
        span = TraceSpan(
            trace_id=self._current_trace_id,
            span_id=str(uuid.uuid4())[:8],
            name=name,
            start_time=time.time(),
            parent_span_id=parent_span_id,
            attributes=attributes or {}
        )
        self.spans.append(span)
        agent_logger.info(
            f"Span started: {name}",
            extra={"extra_fields": {"event": "span_start", "trace_id": span.trace_id, "span_id": span.span_id, "span_name": name, **span.attributes}}
        )
        return span

    def end_span(
        self, span: TraceSpan, status: str = "SUCCESS", error: Optional[Exception] = None, attributes: Optional[Dict[str, Any]] = None
    ):
        span.end_time = time.time()
        span.latency_ms = round((span.end_time - span.start_time) * 1000, 2)
        span.status = status
        if attributes:
            span.attributes.update(attributes)
        if error:
            span.error_message = str(error)
            span.status = "ERROR"

        agent_logger.info(
            f"Span completed: {span.name} [{span.status}] ({span.latency_ms}ms)",
            extra={"extra_fields": {
                "event": "span_end",
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "span_name": span.name,
                "latency_ms": span.latency_ms,
                "status": span.status,
                "error": span.error_message,
                **span.attributes
            }}
        )
        metrics_exporter.record_latency(span.name, span.latency_ms)
        if status == "ERROR":
            metrics_exporter.record_error(span.name, type(error).__name__ if error else "UnknownError")


tracer = Tracer()


class MetricsExporter:
    """
    Aggregates runtime telemetry metrics:
    - Node & API Latencies (avg, p50, p95, max)
    - Error counts by category/node
    - Estimated token usage per LLM invocation
    """

    def __init__(self):
        self.latencies: Dict[str, List[float]] = {}
        self.errors: Dict[str, int] = {}
        self.token_usage: Dict[str, Dict[str, Any]] = {}
        self.cost_threshold_usd = 1.0 # Alert if cost goes above $1.00
        
        # Approximations per 1K tokens for common models (USD)
        self.model_pricing = {
            "gpt-4o": {"prompt": 0.005, "completion": 0.015},
            "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
            "claude-3-5-sonnet-20240620": {"prompt": 0.003, "completion": 0.015},
            "amazon.titan-text-express-v1": {"prompt": 0.0008, "completion": 0.0016},
        }

    def record_latency(self, metric_name: str, latency_ms: float):
        if metric_name not in self.latencies:
            self.latencies[metric_name] = []
        self.latencies[metric_name].append(latency_ms)

    def record_error(self, metric_name: str, error_type: str):
        key = f"{metric_name}:{error_type}"
        self.errors[key] = self.errors.get(key, 0) + 1

    def record_token_usage(self, model: str, prompt_tokens: int, completion_tokens: int):
        if model not in self.token_usage:
            self.token_usage[model] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}
            
        self.token_usage[model]["prompt_tokens"] += prompt_tokens
        self.token_usage[model]["completion_tokens"] += completion_tokens
        self.token_usage[model]["total_tokens"] += (prompt_tokens + completion_tokens)
        
        # Calculate cost
        pricing = self.model_pricing.get(model, {"prompt": 0.0, "completion": 0.0})
        cost = (prompt_tokens / 1000.0) * pricing["prompt"] + (completion_tokens / 1000.0) * pricing["completion"]
        self.token_usage[model]["estimated_cost_usd"] += cost
        
        # Alert mechanism
        total_cost = sum(usage.get("estimated_cost_usd", 0) for usage in self.token_usage.values())
        if total_cost > self.cost_threshold_usd:
            logger.warning(f"🚨 ALERT: LLM API Cost has exceeded the threshold! Current cost: ${total_cost:.4f}")

    def get_summary(self) -> Dict[str, Any]:
        latency_summary = {}
        for name, values in self.latencies.items():
            if values:
                sorted_v = sorted(values)
                p50_idx = int(len(sorted_v) * 0.5)
                p95_idx = min(int(len(sorted_v) * 0.95), len(sorted_v) - 1)
                latency_summary[name] = {
                    "count": len(values),
                    "avg_ms": round(sum(values) / len(values), 2),
                    "p50_ms": round(sorted_v[p50_idx], 2),
                    "p95_ms": round(sorted_v[p95_idx], 2),
                    "max_ms": round(max(values), 2)
                }
        return {
            "latencies": latency_summary,
            "errors": self.errors,
            "token_usage": self.token_usage
        }


metrics_exporter = MetricsExporter()

def trace_node(node_name: str):
    """
    Decorator to trace a langgraph node execution, logging its input, output, latency, and errors.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(state: Dict[str, Any], *args, **kwargs):
            span = tracer.start_span(node_name)
            try:
                result = func(state, *args, **kwargs)
                tracer.end_span(span, status="SUCCESS")
                return result
            except Exception as e:
                tracer.end_span(span, status="ERROR", error=e)
                raise
        return wrapper
    return decorator
