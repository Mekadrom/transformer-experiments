
# Use the official Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main.py file to the working directory
COPY main.py .

# Expose the port on which the server will run
EXPOSE 5000

# Set the entrypoint command to start the server
CMD ["python", "main.py"]
