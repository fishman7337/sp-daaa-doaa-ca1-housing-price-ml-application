FROM python:3.12-slim

# Prevent .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade "pip>=26.1.2,<27" \
    && python -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Keep runtime writes constrained to the instance and upload directories.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/instance /app/static/uploads \
    && chown -R appuser:appuser /app/instance /app/static/uploads

USER appuser

# Default local port; platforms such as Render can override PORT at runtime.
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '5000') + '/', timeout=5)"

# Use shell expansion so PORT/worker settings remain configurable.
CMD ["sh", "-c", "gunicorn wsgi:app --bind 0.0.0.0:${PORT:-5000} --workers ${WEB_CONCURRENCY:-1} --timeout ${WEB_TIMEOUT:-120}"]
