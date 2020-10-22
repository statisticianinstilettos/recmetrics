FROM python:3.7.6-slim

RUN apt-get update -yqq &&\
    apt-get upgrade -yqq &&\
    apt-get install gcc libenchant1c2a -yqq --fix-missing
    
WORKDIR /recmetrics

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.0.10

# System dependencies
RUN pip install "poetry==$POETRY_VERSION"

# Copy Poetry requirements to cache them in Docker layer
COPY poetry.lock pyproject.toml /recmetrics/

# Project initialization
RUN poetry config virtualenvs.create false \
    && poetry install

# Display Pytest output in color
ENV PYTEST_ADDOPTS="--color=yes"

COPY . .