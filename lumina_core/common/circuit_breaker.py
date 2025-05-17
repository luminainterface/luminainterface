from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Deque
from collections import deque
import logging
from prometheus_client import Gauge, Counter

logger = logging.getLogger(__name__)

# Circuit breaker metrics
circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (1=closed, 0=open)",
    ["service", "type"]
)

circuit_breaker_failures = Counter(
    "circuit_breaker_failures_total",
    "Total number of circuit breaker failures",
    ["service", "type"]
)

class CircuitBreaker:
    def __init__(
        self,
        service_name: str,
        service_type: str,
        failure_threshold: int = 5,
        reset_timeout: int = 30,
        success_threshold: int = 2,
        max_failure_history: int = 10
    ):
        self.service_name = service_name
        self.service_type = service_type
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold
        self.failures = 0
        self.successes = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.is_open = False
        self.half_open = False
        self.failure_history: Deque[Dict[str, Any]] = deque(maxlen=max_failure_history)
        self.cached_status: Optional[Dict[str, Any]] = None

    def record_failure(self, error: str):
        """Record a failure and update circuit breaker state."""
        self.failures += 1
        self.last_failure_time = datetime.utcnow()
        self.failure_history.append({
            "timestamp": self.last_failure_time.isoformat(),
            "error": error
        })
        
        circuit_breaker_failures.labels(
            service=self.service_name,
            type=self.service_type
        ).inc()
        
        if self.failures >= self.failure_threshold:
            self.is_open = True
            self.half_open = False
            self.successes = 0
            circuit_breaker_state.labels(
                service=self.service_name,
                type=self.service_type
            ).set(0)
            logger.warning(f"Circuit breaker opened for {self.service_name}")

    def record_success(self):
        """Record a success and update circuit breaker state."""
        if self.is_open:
            self.successes += 1
            if self.successes >= self.success_threshold:
                self.is_open = False
                self.half_open = False
                self.failures = 0
                self.successes = 0
                circuit_breaker_state.labels(
                    service=self.service_name,
                    type=self.service_type
                ).set(1)
                logger.info(f"Circuit breaker closed for {self.service_name}")
        else:
            self.failures = max(0, self.failures - 1)  # Decay failures
        
        self.last_success_time = datetime.utcnow()

    def can_try(self) -> bool:
        """Check if a request can be attempted."""
        if not self.is_open:
            return True
        
        if self.last_failure_time and \
           (datetime.utcnow() - self.last_failure_time).total_seconds() > self.reset_timeout:
            self.half_open = True
            return True
        
        return False

    def get_cached_status(self) -> Optional[Dict[str, Any]]:
        """Get cached status if available and not expired."""
        if self.cached_status and self.last_success_time:
            if (datetime.utcnow() - self.last_success_time) < timedelta(minutes=5):
                return self.cached_status
        return None

    def update_cache(self, status: Dict[str, Any]):
        """Update the cached status."""
        self.cached_status = status
        self.last_success_time = datetime.utcnow()

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "is_open": self.is_open,
            "half_open": self.half_open,
            "failures": self.failures,
            "successes": self.successes,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "failure_history": list(self.failure_history)
        } 