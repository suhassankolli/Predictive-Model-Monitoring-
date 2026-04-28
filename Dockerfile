FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY modelsentinel/ modelsentinel/

RUN pip install --no-cache-dir -e ".[vertex]"

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PYSPARK_PYTHON=python3

ENTRYPOINT ["python", "-m", "modelsentinel.runner"]
