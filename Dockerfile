FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer stays cached between code changes.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Files used when the pipeline or the web console runs.
COPY main.py config.yaml ./
COPY frontend ./frontend

# Run as a non-root user with the folders the pipeline writes to.
RUN useradd --create-home appuser \
    && mkdir -p data/raw data/interim data/processed \
                artifacts/models artifacts/reports logs \
    && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
