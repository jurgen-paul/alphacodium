# Second stage: Python 3.11
FROM python:3.11.0-slim as python311

# Final stage: Ubuntu base image
FROM python:3.9.12-slim

# Avoid interactive dialogue with tzdata when installing packages
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python installations from the previous stages
COPY --from=python311 /usr/local /usr/local/python3.11

RUN python --version
RUN pip --version


COPY requirements.txt .

RUN pip install -r requirements.txt

ENV PYTHON311BIN=/root/.pyenv/versions/3.11.0/bin/python
ENV PYTHON311LIB=/root/.pyenv/versions/3.11.0/lib/python3.11

