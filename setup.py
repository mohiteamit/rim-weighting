from setuptools import setup, find_packages

setup(
    name="rim_weighting",
    version="0.1.7",
    author="Amit Mohite",
    author_email="amit.mohite@outlook.com",
    description="A Python implementation of the Random Iterative Method (RIM) weighting algorithm for survey data adjustment.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mohiteamit/rim-weighting",
    packages=find_packages(include=["rim_weighting", "rim_weighting.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas",
        "numpy",
        "pytest",
        "tabulate",
        "pyspark"
    ],
    include_package_data=True,
    package_data={
        "examples": ["examples/*.csv"],
    },
)
