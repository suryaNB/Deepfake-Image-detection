FROM python:3.10-slim

# Install system deps needed for dlib, face_recognition, opencv, TF wheels
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

WORKDIR /app

# Copy project
COPY . .

# Upgrade pip/build tools
RUN pip install --upgrade pip setuptools wheel

# Install Python deps
RUN pip install -r requirements.txt

# Expose port (optional for readability)
EXPOSE 8080

# Use Render's PORT environment variable (Render sets PORT automatically)
CMD ["bash", "-lc", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
