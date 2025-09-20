# --- Stage 1: The "Builder" ---
# This stage installs build tools and compiles our Python dependencies.
FROM python:3.9-slim AS builder

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set up a virtual environment to cleanly isolate packages
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy and install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: The Final "Runtime" Image ---
# This stage is our final, lightweight application image.
FROM python:3.9-slim

# Install ONLY the necessary runtime system libraries for OpenCV and Paddle
# This fixes the "libgthread-2.0.so.0" error
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the virtual environment (with all installed Python packages) from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate the virtual environment for subsequent commands
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application code
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# The command to run when the container starts
ENTRYPOINT ["streamlit", "run", "app.py"]