FROM python:3.9-slim-bullseye


COPY requirements_app.txt /app/
RUN pip install --upgrade pip
RUN pip install -r app/requirements_app.txt

COPY templates /app/
COPY app.py /app/

WORKDIR /app

ENTRYPOINT [ "python" ]
CMD ["app.py"]