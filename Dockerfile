# Use the official Python image from the Docker Hub
FROM python:3.11-slim-buster

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y gcc g++ pkg-config libmariadb-dev-compat && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libgl1-mesa-glx


# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container



# Run Django with Gunicorn
CMD ["gunicorn", "--bind", "127.0.0.1:8000", "myproject.wsgi:application"]


#dsfsdfds