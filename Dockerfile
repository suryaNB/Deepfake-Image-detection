# Use Python 3.10 (required for dlib + TensorFlow)
FROM python:3.10-slim

# Install system dependencies needed for dlib, face_recognition, TensorFlow, OpenCV, numpy
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    make \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project code
COPY . .

# Upgrade pip, setuptools, wheel (required for building dlib/numpy)
RUN pip install --upgrade pip setuptools wheel

# Install project dependencies
RUN pip install -r requirements.txt

# Expose port (Railway uses 8080 internally)
EXPOSE 8080

# Start Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
