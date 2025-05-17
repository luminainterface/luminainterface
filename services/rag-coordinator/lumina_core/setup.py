from setuptools import setup, find_packages

setup(
    name="lumina_core",
    version="0.1.0",
    packages=find_packages(include=["lumina_core", "lumina_core.*"]),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.4.2",
        "pydantic-settings>=2.0.0",
        "python-dotenv>=1.0.0",
        "redis>=5.0.1",
        "qdrant-client>=1.6.0",
        "aiohttp>=3.9.0",
        "prometheus-client>=0.19.0",
        "apscheduler>=3.10.4",
        "pytest>=7.4.3",
        "httpx>=0.25.0",
        "python-multipart>=0.0.6",
        "fastapi-limiter>=0.1.5",
        "opentelemetry-api==1.21.0",
        "opentelemetry-sdk==1.21.0",
        "opentelemetry-instrumentation-fastapi==0.42b0",
        "opentelemetry-exporter-otlp==1.21.0",
        "opentelemetry-exporter-jaeger-proto-grpc==1.21.0",
        "opentelemetry-exporter-jaeger-thrift==1.21.0",
        "opentelemetry-instrumentation-redis==0.42b0",
        "opentelemetry-instrumentation-httpx==0.42b0"
    ],
) 