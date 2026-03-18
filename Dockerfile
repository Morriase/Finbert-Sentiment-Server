FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set PORT environment variable (Render sets this automatically, but we provide a default)
ENV PORT=8000
EXPOSE $PORT

# Start the uvicorn server stringing with PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
