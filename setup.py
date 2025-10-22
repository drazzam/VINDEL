"""
VINDEL Package Setup
VINe-based DEgree-of-freedom Learning
An LLM-Integrated Framework for Synthetic IPD Generation
"""

import os
from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="VINDEL",
    author="Ahmed Y. Azzam",
    author_email="ahmed.azzam@hsc.wvu.edu",  
    description="LLM-Integrated Bayesian Framework for Synthetic Individual Patient Data Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vindel",  # Add if desired
    project_urls={
        "Documentation": "https://github.com/drazzam/VINDEL", # Check README.md file
        "Source": "https://github.com/drazzam/VINDEL",
    },
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "synthetic-data",
        "bayesian-statistics",
        "survival-analysis",
        "vine-copulas",
        "healthcare",
        "clinical-trials",
        "meta-analysis",
        "individual-patient-data",
        "evidence-synthesis",
        "bayesian-model-averaging",
        "llm-integration",
        "claude-ai"
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "lifelines>=0.27.0",
        "pyvinecopulib>=0.6.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    package_data={
        "vindel": [
            "py.typed",  # PEP 561 compliance
        ],
    },
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "vindel-validate=vindel.utils.validation:main",  # Could add CLI tools
        ],
    },
)
