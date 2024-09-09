# Use the official Python runtime as a parent image
FROM python:3.9.7-slim

# Set the working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONPATH=/app

# Run the application
CMD ["python", "-m", "pinngnn.PEMS04"]  # Or other Python module you want to run