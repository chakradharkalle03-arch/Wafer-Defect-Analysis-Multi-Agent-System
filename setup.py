"""
Setup script for Wafer Defect Analysis System
"""
from setuptools import setup, find_packages

setup(
    name="wafer-defect-analysis",
    version="1.0.0",
    description="AI-powered multi-agent system for wafer defect analysis",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "langgraph>=0.0.20",
        "langchain>=0.1.0",
        "ultralytics>=8.1.0",
        "opencv-python>=4.8.1.78",
        "pillow>=10.1.0",
        "torch>=2.1.1",
        "torchvision>=0.16.1",
        "transformers>=4.36.0",
        "huggingface-hub>=0.19.4",
        "numpy>=1.24.3",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.0",
        "pandas>=2.1.4",
        "reportlab>=4.0.7",
        "httpx>=0.25.2",
    ],
    python_requires=">=3.8",
)

