from setuptools import setup, find_packages
    
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
setup(
    name="zenith-nlp-framework",
    version="1.0.0",
    author="Satya Sai Nischal",
    author_email="coderstale@gmail.com",
    description="An advanced, from-scratch NLP framework for training and deploying modern transformer models.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cattolatte/zenith-nlp-framework",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
            "Intended Audience :: Developers",            "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.9',
)
    

