# setup.py
from setuptools import setup, find_packages

setup(
    name="stats_library",
    version="0.1.0",
    author="Your Name",
    description="A simple stats library with trees and forests",
    packages=find_packages(exclude=["tests", "docs"]),  
    install_requires=[
    ],
    python_requires=">=3.8",
)
