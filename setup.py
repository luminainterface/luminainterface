from setuptools import setup, find_packages

setup(
    name="lumina_core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "redis>=4.0.0",
        "qdrant-client>=1.1.0",
        "websockets>=10.0",
        "python-dotenv>=0.19.0",
        "requests>=2.26.0",
        "pydantic>=1.8.0",
        "aiohttp>=3.8.0",
        "prometheus-client>=0.12.0",
    ],
) 