<div align="center">


███ ███ █  █ ███ ███ █ █ █  █ █   ██  
  █ █   ██ █  █   █  █ █ ██ █ █   █ █ 
 █  ██  █ ██  █   █  ███ █ ██ █   ██  
█   █   █ ██  █   █  █ █ █ ██ █   █   
█   █   █  █  █   █  █ █ █  █ █   █   
███ ███ █  █ ███  █  █ █ █  █ ███ █   
                                    

### A Framework for Advanced Natural Language Processing
</div>

[![PyPI version](https://badge.fury.io/py/my-nlp-framework.svg)](https://badge.fury.io/py/my-nlp-framework)
[![Python versions](https://img.shields.io/pypi/pyversions/my-nlp-framework.svg)](https://pypi.org/project/my-nlp-framework)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ZenithNLP** is an advanced, from-scratch NLP framework built with PyTorch for training, fine-tuning, and deploying modern transformer-based models. It serves as a comprehensive toolkit for NLP practitioners and researchers, featuring a modular architecture and a full suite of MLOps capabilities.

## ✨ Features

- **State-of-the-Art Model Architectures**: From-scratch implementations of:
  - `BERT` (Encoder-only) for tasks like classification and NER.
  - `GPT` (Decoder-only) for causal language modeling and text generation.
  - `Seq2SeqTransformer` (Encoder-Decoder) for translation and summarization.
- **Advanced Training Techniques**:
  - **Parameter-Efficient Fine-Tuning (PEFT)**: Integrated **LoRA** (Low-Rank Adaptation) for efficient fine-tuning of large models.
  - **Distributed Training**: Support for multi-GPU training using PyTorch's `DistributedDataParallel`.
  - **Advanced Optimization**: Includes learning rate scheduling with warm-up and gradient clipping.
- **Full MLOps Pipeline**:
  - **Configuration Management**: Powered by **Hydra**, allowing for flexible and reproducible experiments through YAML files.
  - **Experiment Tracking**: Integrated with **MLflow** to log parameters, metrics, and model artifacts automatically.
  - **Containerization**: Fully containerized with **Docker** and **Docker Compose** for reproducible environments and easy deployment of the MLflow UI.
  - **Continuous Integration**: Automated testing pipeline with **GitHub Actions** and `pytest`.
- **Flexible API for Deployment**:
  - A ready-to-use **FastAPI** server that can dynamically load and serve any model trained with the framework.
- **Custom Core Components**:
  - A trainable **Byte-Pair Encoding (BPE) Tokenizer** built from scratch.
  - Modular implementations of `MultiHeadAttention`, `PositionalEncoding`, and other core transformer building blocks.

## 🚀 Getting Started

### 1. Installation (from PyPI)

> **Note**: Once published, you will be able to install the framework directly from PyPI.

```bash
pip install zenith-nlp-framework
```

### 2. Local Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/cattolatte/zenith-nlp-framework.git
cd zenith-nlp-framework

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Install the project in editable mode
pip install -e .
```

## 📖 Tutorial: Training a Text Classifier

This framework is designed for flexibility. Here’s how you can train your own text classification model.

### 1. Prepare Your Data and Configs
Place your training data (e.g., `my_data.csv`) in a local `data/` directory. Use the `configs/` directory as a template. You can modify `config.yaml` or create a new one to point to your data file and adjust model/training parameters.

### 2. Run Training
Run the text classification task script. All parameters are managed by the Hydra configuration files in the `configs/` directory.

```bash
# Run with default settings from the config files
python3 -m my_nlp_framework.tasks.text_classification
```

You can easily override any parameter from the command line:

```bash
# Train for more epochs with a different learning rate
python3 -m my_nlp_framework.tasks.text_classification training.epochs=10 training.learning_rate=0.0005

# Train with LoRA enabled
python3 -m my_nlp_framework.tasks.text_classification model.use_lora=True model.lora_rank=8
```

### 3. Track Experiments with MLflow
Before training, launch the MLflow UI to track your experiments in real-time. The `docker-compose.yml` file is pre-configured for you.

```bash
# Start the MLflow server in the background
docker-compose up -d
```
Navigate to **http://localhost:5000** in your browser to view the MLflow dashboard.

🌐 Serving Your Model via API
Once you have a trained model (`.pth` file) and tokenizer (`.json` file), you can easily deploy it with the built-in FastAPI server.

```bash
python3 -m my_nlp_framework.inference.api \
    --model-path /path/to/your/trained_model.pth \
    --tokenizer-path /path/to/your/tokenizer.json \
    --vocab-size 10000 \
    --num-classes 2
```

The API will be available at **http://localhost:8000/docs** for interactive testing.

🐳 Running with Docker
You can also run the entire training process within a Docker container for perfect reproducibility.

```bash
# 1. Build the Docker image
docker build -t zenith-nlp-framework:latest .

# 2. Run a task (mounting your local data directory)
docker run --rm -v "$(pwd)/data":/app/data zenith-nlp-framework:latest \
  python -m my_nlp_framework.tasks.text_classification
```

🏛️ Framework Architecture
This framework is organized into several key modules:

`src/my_nlp_framework/core`: Contains the fundamental building blocks like attention mechanisms, LoRA layers, and tokenizers.

`src/my_nlp_framework/models`: Defines high-level model architectures like BERT and GPT.

`src/my_nlp_framework/data`: Includes flexible data loaders.

`src/my_nlp_framework/training`: A powerful, centralized training engine with advanced features.

`src/my_nlp_framework/tasks`: Example scripts that show how to use the framework to solve end-to-end problems.

`src/my_nlp_framework/inference`: Code for deploying and serving trained models.

`configs/`: Centralized YAML configuration files for Hydra.

`tests/`: Unit and integration tests for the framework.

This project was built as an advanced, from-scratch implementation of a modern NLP framework.
