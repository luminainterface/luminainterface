from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXInstrumentor
from fastapi import FastAPI

from lumina_core.common.config import get_settings

def setup_tracing(app: FastAPI, service_name: str) -> None:
    """Configure OpenTelemetry tracing for a FastAPI application."""
    settings = get_settings()
    
    if not settings.TRACING_ENABLED:
        return
    
    # Configure tracer
    tracer_provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )
    
    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=settings.JAEGER_AGENT_HOST,
        agent_port=settings.JAEGER_AGENT_PORT,
    )
    
    # Add span processor to the tracer
    tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
    
    # Set the tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    
    # Instrument Redis
    RedisInstrumentor().instrument()
    
    # Instrument HTTPX for outgoing requests
    HTTPXInstrumentor().instrument()

def create_span(name: str, context: dict = None) -> trace.Span:
    """Create a new trace span with optional context."""
    tracer = trace.get_tracer(__name__)
    span = tracer.start_span(name)
    
    if context:
        for key, value in context.items():
            span.set_attribute(key, value)
    
    return span 