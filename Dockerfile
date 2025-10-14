# 1. Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies required for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application's source code
COPY . .

# 6. Install the project in editable mode
RUN pip install -e .

# The container can now be run with a command like:
# docker run --rm my_nlp_framework python -m my_nlp_framework.tasks.text_classification