FROM python:3.12-slim

WORKDIR /app

# Install build dependencies needed for compiling C extensions (like hiredis)
# Then install Python packages and remove build tools in the same layer to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first (for better layer caching)
COPY requirements.txt .

# Install dependencies and remove build tools in the same RUN command to reduce layers
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y build-essential gcc && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy only necessary application code (excludes files in .dockerignore)
COPY . .

# Create directory for runtime data (like waiting_list.json)
RUN mkdir -p /app/data && chmod 755 /app/data

EXPOSE 8000

# Use exec form for better signal handling
CMD ["uvicorn", "services.unified_app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
