# Build from repository root:
#   docker build -f tripletex/Dockerfile -t tripletex-agent .
FROM python:3.11-slim

WORKDIR /app

COPY tripletex/requirements.txt /app/tripletex/requirements.txt
RUN pip install --no-cache-dir -r /app/tripletex/requirements.txt

COPY tripletex/ /app/tripletex/

ENV PYTHONPATH=/app
ENV PORT=8080
EXPOSE 8080

CMD sh -c "exec uvicorn tripletex.main:app --host 0.0.0.0 --port ${PORT:-8080}"
