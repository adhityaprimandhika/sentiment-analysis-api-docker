FROM python:3.9-slim

EXPOSE 5000

WORKDIR /app

COPY requirements.txt /app
RUN pip install --user --upgrade pip
RUN pip install --no-cache-dir --user -r requirements.txt

COPY app.py /app
COPY saved-model /app
CMD ["python", "app.py"]