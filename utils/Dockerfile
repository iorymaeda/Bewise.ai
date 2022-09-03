FROM python:3.10

WORKDIR /app

ADD requirements.txt requirements.txt
ADD utils/download_hface.py download_hface.py
ADD api.py api.py

RUN python -m pip install -r requirements.txt
RUN python download_hface.py