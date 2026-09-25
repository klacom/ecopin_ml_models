FROM python:3.11-slim

# Copy the AWS Lambda Web Adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Environment variable to tell the adapter what port the app listens on
# The server is hardcoded to listen on PORT from env or 8002 by default
ENV PORT=8002

# Command to run the Python built-in server
CMD ["python", "inference_service.py"]
