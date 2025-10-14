from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="zenith-nlp-framework",
    version="0.1.0",
    author="Satya Sai Nischal",
    author_email="coderstale@gmail.com",
    description="An advanced NLP framework",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)