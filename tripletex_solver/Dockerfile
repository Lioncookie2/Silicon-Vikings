FROM python:3.11-slim

WORKDIR /app

# Installer dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopier kildekoden
COPY ./app /app/app

# Sett opp for Cloud Run (port 8080 er standard)
ENV PORT=8080
EXPOSE 8080

# Kjør applikasjonen
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
