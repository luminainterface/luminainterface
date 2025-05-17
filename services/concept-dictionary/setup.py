from setuptools import setup, find_packages

setup(
    name="concept-dictionary",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "redis>=4.0.0",
        "qdrant-client>=1.1.0",
        "neo4j>=5.0.0",
        "numpy>=1.21.0",
        "pydantic>=1.8.0",
        "python-dotenv>=0.19.0",
    ],
    python_requires=">=3.8",
) 