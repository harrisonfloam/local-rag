FROM python:3.12

WORKDIR /local-rag

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "app.api.main"]