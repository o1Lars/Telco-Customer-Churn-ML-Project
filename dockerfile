# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage caching
COPY requirements.txt .

# Install dependencies (cached unless requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the source code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Command to run
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
