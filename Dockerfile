# Use Python 3.10 as the base to ensure compatibility with your TensorFlow models
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your files
COPY . .

# Set the default command to run your app
CMD ["python", "app.py"]