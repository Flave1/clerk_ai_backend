FROM python:3.12-slim

WORKDIR /app

# Copy only the requirements first (for layer caching)
COPY requirements.txt .

# Install dependencies and KEEP cache for faster rebuilds
RUN pip install -r requirements.txt

# Now copy the rest of the code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
