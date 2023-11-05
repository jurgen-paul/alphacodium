
FROM python:3.9.12-slim

RUN python --version
RUN pip --version


COPY requirements.txt .

RUN pip install -r requirements.txt

ENV PYTHON311BIN=/root/.pyenv/versions/3.11.0/bin/python
ENV PYTHON311LIB=/root/.pyenv/versions/3.11.0/lib/python3.11

