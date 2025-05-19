from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry

# Initialize metrics registry
registry = CollectorRegistry()

# Service health metrics
service_health = Gauge(
    "service_health",
    "Service health status (1=healthy, 0.5=degraded, 0=unhealthy)",
    ["service", "type", "check"],
    registry=registry
)

service_latency = Histogram(
    "service_latency_seconds",
    "Service latency in seconds",
    ["service", "type", "check"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=registry
)

health_check_errors = Counter(
    "health_check_errors_total",
    "Total health check errors",
    ["service", "type", "check", "error_type"],
    registry=registry
)

health_check_retries = Counter(
    "health_check_retries_total",
    "Total health check retries",
    ["service", "type", "check"],
    registry=registry
)

# System metrics
system_metrics = Gauge(
    "system_metrics",
    "System resource usage",
    ["resource", "type"],
    registry=registry
)

# Infrastructure metrics
infrastructure_health = Gauge(
    "infrastructure_health",
    "Infrastructure component health status",
    ["component", "type"],
    registry=registry
)

infrastructure_latency = Histogram(
    "infrastructure_latency_seconds",
    "Infrastructure component latency",
    ["component", "type"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
    registry=registry
)

# Business metrics
business_health = Gauge(
    "business_health",
    "Business component health status",
    ["component", "type"],
    registry=registry
)

business_latency = Histogram(
    "business_latency_seconds",
    "Business component latency",
    ["component", "type"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
    registry=registry
)

# Required metrics for health checks
REQUIRED_METRICS = [
    "service_health",
    "service_latency_seconds_count",
    "health_check_errors_total",
    "health_check_retries_total",
    "system_metrics",
    "infrastructure_health",
    "business_health"
]

def validate_metrics(metrics_text: str) -> bool:
    """Validate that all required metrics are present."""
    return all(metric in metrics_text for metric in REQUIRED_METRICS) 