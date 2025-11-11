FROM python:3.10.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

# Crea y activa venv en /opt/venv
ENV VENV=/opt/venv
RUN python -m venv "$VENV" && \
  "$VENV/bin/pip" install --upgrade pip setuptools wheel
ENV PATH="$VENV/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  tini \
  && rm -rf /var/lib/apt/lists/*

# Trabajo en /app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia TODO el repo (incluye app/, reports/, results/, data/, images/, etc.)
COPY . .

# Config Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true \
  STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Puerto de Streamlit
EXPOSE 8501

# Entrypoint simple y estable
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
