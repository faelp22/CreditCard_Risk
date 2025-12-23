# ==============================================================================
# Multi-stage build para imagem mínima de produção
# ==============================================================================

# -----------------------
# Stage 1: Builder
# -----------------------
FROM python:3.11-slim AS builder

# Instalar dependências de sistema necessárias para build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /build

# Copiar apenas requirements para aproveitar cache do Docker
COPY requirements.txt .

# Instalar dependências Python em um diretório isolado
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# -----------------------
# Stage 2: Runtime
# -----------------------
FROM python:3.11-slim

# Metadados da imagem
LABEL maintainer="Isael <liakruk@github>"
LABEL description="Credit Card Risk Analysis - Streamlit Application"
LABEL version="1.0"

# Variáveis de ambiente para otimização
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

# Criar usuário não-privilegiado
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1001 -m -s /sbin/nologin appuser

# Definir diretório de trabalho
WORKDIR /app

# Copiar dependências Python do builder
COPY --from=builder /install /usr/local

# Copiar código-fonte e modelo treinado
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser data/UCI_Credit_Card.csv ./data/UCI_Credit_Card.csv

# Criar diretório para configurações do Streamlit
RUN mkdir -p /app/.streamlit && \
    chown -R appuser:appuser /app

# Configuração do Streamlit
COPY --chown=appuser:appuser <<EOF /app/.streamlit/config.toml
[server]
headless = true
port = 8501
address = "0.0.0.0"
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[client]
showErrorDetails = false
toolbarMode = "minimal"

[theme]
primaryColor = "#FF4B4B"

[logger]
level = "info"
EOF

# Mudar para usuário não-privilegiado
USER appuser

# Expor porta do Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')" || exit 1

# Comando para iniciar a aplicação
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# docker build -t faelp22/credit-risk-analysis:latest .
# docker build -t faelp22/credit-risk-analysis:latest . && docker compose up -d
