FROM python:3.11-slim

WORKDIR /app

# Install Chromium and other dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libwayland-client0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    git \
    chromium \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies in a specific order to avoid conflicts
COPY requirements.txt .

# Install core dependencies first
RUN pip install --no-cache-dir fastapi uvicorn pydantic sse-starlette python-multipart redis aioredis boto3

# Install langchain-core at the specific version required by browser-use
RUN pip install --no-cache-dir langchain-core==0.3.49

# Install browser dependencies
RUN pip install --no-cache-dir playwright==1.51.0 patchright==1.51.3

# Install langchain packages with --no-deps to avoid conflicts
RUN pip install --no-cache-dir --no-deps langchain==0.3.21 langchain-openai==0.3.11

# Install browser-use with --no-deps to avoid dependency conflicts
RUN pip install --no-cache-dir --no-deps git+https://github.com/neo773/browser-use.git@bot-detection-batch

# Create directories for logs and states
RUN mkdir -p /app/logs/states /app/logs/videos

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PLAYWRIGHT_BROWSERS_PATH=0
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
ENV CHROMIUM_PATH=/usr/bin/chromium
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
