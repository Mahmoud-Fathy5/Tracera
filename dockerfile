FROM python:3.10-slim

# HF Spaces runs as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY --chown=user . .

# HF Spaces REQUIRES port 7860
EXPOSE 7860

# Run with gunicorn (already in your requirements.txt)
CMD ["gunicorn", "app:create_app()", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--timeout", "120", \
     "--preload"]