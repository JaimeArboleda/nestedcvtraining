FROM python:3.8-slim-buster

WORKDIR /nestedcv

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN pip3 install notebook

COPY . .