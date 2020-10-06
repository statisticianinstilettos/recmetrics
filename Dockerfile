FROM python:3.7.6-slim

RUN apt-get update -yqq &&\
    apt-get upgrade -yqq &&\
    apt-get install gcc libenchant1c2a -yqq --fix-missing
    
WORKDIR /recmetrics

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .