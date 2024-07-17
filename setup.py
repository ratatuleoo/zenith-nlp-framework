from setuptools import setup, find_packages

setup(
    name="my_nlp_framework",
    version="0.1.0",
    author="Satya Sai Nischal",
    author_email="coderstale@gmail.com",
    description="An advanced NLP framework",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "nltk",
        "spacy",
        "transformers",
        "torch",
        "tensorflow",
        "flask",
        "matplotlib",
        "seaborn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
