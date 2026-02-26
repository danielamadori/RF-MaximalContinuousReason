# MCR
FROM python:3.11-slim

LABEL maintainer="MCR Project"
LABEL description="MCR Container"

WORKDIR /app

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    redis-server \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copia e installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia il codice dell'applicazione
COPY . .

# Crea le directory necessarie
RUN mkdir -p logs workers fig results /var/log/supervisor

# Copia configurazione supervisor
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Espone porte
EXPOSE 6379
EXPOSE 8888

# Variabili d'ambiente
ENV PYTHONUNBUFFERED=1 \
    REDIS_HOST=localhost \
    REDIS_PORT=6379 \
    JUPYTER_ENABLE_LAB=yes

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD redis-cli ping || exit 1

# Avvia supervisor per gestire Redis
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

