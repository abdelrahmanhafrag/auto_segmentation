# Multi-stage build for production-ready PET Segmentation container
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=true \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Build stage
FROM base as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY pet_segmentation/ ./pet_segmentation/
COPY auto_segmenter.py ./
COPY pyproject.toml README.md ./

# Install the package
RUN pip install --no-deps -e .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app

# Create necessary directories
RUN mkdir -p /app/outputs /app/temp /app/logs && \
    chown -R app:app /app/outputs /app/temp /app/logs

USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pet_segmentation; print('Health check passed')" || exit 1

# Default command
CMD ["python", "-m", "pet_segmentation.cli.main", "--help"]

# Development stage
FROM production as development

USER root

# Install development dependencies
RUN pip install pytest pytest-cov black flake8 mypy pre-commit

# Install development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

USER app

# Set development environment
ENV DEBUG=true \
    LOG_LEVEL=DEBUG

CMD ["python", "-m", "pet_segmentation.cli.main"] 