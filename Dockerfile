# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code
COPY . .

# Expose the port the app runs on
# Koyeb will set the PORT environment variable, which we'll use in the CMD
EXPOSE 7275

# Define the command to run your app
# We use the PORT environment variable provided by Koyeb, defaulting to 7275
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7275}
