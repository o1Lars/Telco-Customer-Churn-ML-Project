FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (better layer caching)
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy entire project
COPY . .

# Ensure Python can import from /app/src
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]