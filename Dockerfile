FROM python:3.10

WORKDIR /app

ADD utils/ utils/
ADD utils/download_hface.py download_hface.py
ADD api.py api.py
ADD requirements.txt requirements.txt

RUN python -m pip install -r requirements.txt
RUN python download_hface.py

CMD [ "gunicorn", \
    # FastAPI using uvicorn as ASGI framework
    # Swith worker class to Uvicorn to avoid errors
    "-k", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:8200", "api:app" \
    ]