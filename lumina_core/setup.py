from setuptools import setup, find_packages

setup(
    name="lumina_core",
    version="0.1.0",
    packages=find_packages(include=["lumina_core", "lumina_core.*", "lumina_core.common", "lumina_core.common.*"]),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.4.2",
        "python-dotenv>=1.0.0",
        "redis>=5.0.1",
        "qdrant-client>=1.6.0",
        "aiohttp>=3.9.0",
        "prometheus-client>=0.19.0",
        "apscheduler>=3.10.4",
        "pytest>=7.4.3",
        "httpx>=0.25.0",
        "python-multipart>=0.0.6",
        "fastapi-limiter>=0.1.5"
    ],
) 