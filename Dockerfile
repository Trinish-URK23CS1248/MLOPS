# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements (if any)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all source code + artifacts
COPY . .

# Ensure artifacts directory exists and model is copied
RUN mkdir -p /app/artifacts
COPY artifacts/model.pkl /app/artifacts/model.pkl

# Expose port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
