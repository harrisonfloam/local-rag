FROM python3.12-bookworm

WORKDIR /local-rag

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "app.api.main"]