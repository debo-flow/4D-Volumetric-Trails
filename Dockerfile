# Base image with Python 3.10 and PyTorch (CUDA supported for GPU)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, UI, and COLMAP
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    colmap \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose port for Gradio Web Viewer
EXPOSE 7860

# Default command to run when the container starts
CMD ["python", "src/web_viewer.py", "--model", "data/output/spacetime_trail.glb"]
