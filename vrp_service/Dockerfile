FROM python:3.11-slim

# Copy the AWS Lambda Web Adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY vrp_solver.py .

# Environment variable to tell the adapter what port the app listens on
ENV PORT=8003

# Command to run the Flask app using gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8003", "vrp_solver:app"]
