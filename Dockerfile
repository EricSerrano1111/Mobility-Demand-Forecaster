# Use a lightweight Python base image
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr (crucial for GCP logging)
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy ONLY the API requirements first (to leverage Docker layer caching)
COPY api/requirements.txt .

# Install the production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary directories into container
# Only bring API script and trained model, leaving raw data and notebooks behind
COPY api/ /app/api/
COPY models/ /app/models/

# Expose the port Google Cloud Run defaults
EXPOSE 8080

# Command to boot the server when the container starts
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]