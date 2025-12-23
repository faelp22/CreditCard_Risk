# 🐳 Guia de Deploy com Docker

## 📋 Pré-requisitos

- **Docker** >= 20.10
- **Docker Compose** >= 2.0
- Modelo treinado em `models/modelo_credito.pkl`

---

## 🚀 Quick Start

### 1️⃣ Build da Imagem

```bash
docker-compose build
```

### 2️⃣ Iniciar Aplicação

```bash
docker-compose up -d
```

### 3️⃣ Acessar Aplicação

Abra no navegador: **http://localhost:8501**

### 4️⃣ Parar Aplicação

```bash
docker-compose down
```

---

## 🔧 Comandos Úteis

### Ver Logs em Tempo Real
```bash
docker-compose logs -f streamlit-app
```

### Verificar Status
```bash
docker-compose ps
```

### Reiniciar Aplicação
```bash
docker-compose restart
```

### Remover Tudo (incluindo volumes)
```bash
docker-compose down -v
```

### Build sem Cache
```bash
docker-compose build --no-cache
```

---

## 🏗️ Arquitetura da Imagem

### **Multi-Stage Build**
- **Stage 1 (Builder)**: Compila dependências Python
- **Stage 2 (Runtime)**: Imagem mínima apenas com o necessário

### **Otimizações Implementadas**

✅ **Segurança**
- Usuário não-privilegiado (`appuser` UID 1001)
- Filesystem em modo somente leitura
- Sem privilégios adicionais (`no-new-privileges`)
- XSRF Protection habilitado

✅ **Performance**
- Imagem baseada em `python:3.11-slim` (~150-200 MB)
- Cache de layers otimizado
- Dependências pré-compiladas

✅ **Produção**
- Health check automático
- Logs estruturados (JSON)
- Restart automático em caso de falha
- Limites de recursos configuráveis

---

## 📦 Estrutura de Arquivos Incluídos na Imagem

```
/app/
├── src/
│   ├── app.py              # Aplicação Streamlit
│   ├── train_model.py      # Script de treinamento
│   ├── llm.py              # Integração LLM
│   ├── explain.py          # Explicabilidade (SHAP)
│   └── prompts.py          # Templates de prompts
│
├── models/
│   └── modelo_credito.pkl  # Modelo treinado (incluído!)
│
├── data/
│   └── UCI_Credit_Card.csv # Dataset de exemplo
│
└── .streamlit/
    └── config.toml         # Configurações do Streamlit
```

---

## ⚙️ Variáveis de Ambiente

Você pode customizar o comportamento editando o `docker-compose.yml`:

```yaml
environment:
  - STREAMLIT_SERVER_PORT=8501              # Porta do servidor
  - STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200    # Tamanho máximo upload (MB)
  - STREAMLIT_SERVER_ENABLE_CORS=false      # CORS (desabilitado por segurança)
  - PYTHONUNBUFFERED=1                      # Logs em tempo real
```

---

## 🔒 Segurança

### **Recursos de Segurança Implementados**

1. **Usuário Não-Privilegiado**
   - Aplicação roda como `appuser` (UID 1001)
   - Sem acesso root

2. **Filesystem Somente Leitura**
   - Sistema de arquivos protegido contra modificações
   - Apenas `/tmp` e cache do Streamlit são graváveis

3. **Limites de Recursos**
   - CPU: 2 cores (máximo), 0.5 cores (reservado)
   - RAM: 2GB (máximo), 512MB (reservado)

4. **Health Check**
   - Verifica saúde da aplicação a cada 30s
   - Reinicia automaticamente em caso de falha

---

## 🎯 Cenários de Uso

### **Desenvolvimento Local**
```bash
docker-compose up
```

### **Produção (Background)**
```bash
docker-compose up -d
```

### **Teste Rápido**
```bash
docker run -p 8501:8501 creditcard-risk:latest
```

### **Build Manual da Imagem**
```bash
docker build -t creditcard-risk:latest .
```

---

## 📊 Monitoramento

### **Ver Uso de Recursos**
```bash
docker stats creditcard-risk-app
```

### **Inspecionar Container**
```bash
docker inspect creditcard-risk-app
```

### **Health Check Status**
```bash
docker inspect --format='{{.State.Health.Status}}' creditcard-risk-app
```

---

## 🐛 Troubleshooting

### **Erro: Porta 8501 já está em uso**
```bash
# Verificar processo usando a porta
netstat -ano | findstr :8501  # Windows
lsof -i :8501                 # Linux/Mac

# Mudar a porta no docker-compose.yml
ports:
  - "8502:8501"
```

### **Erro: Modelo não encontrado**
```bash
# Verificar se o modelo existe
ls -lh models/modelo_credito.pkl

# Treinar o modelo se necessário
cd src && python train_model.py
```

### **Erro: Out of Memory**
```bash
# Aumentar limite de memória no docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
```

### **Logs Detalhados**
```bash
# Ver últimas 100 linhas
docker-compose logs --tail=100 streamlit-app

# Seguir logs em tempo real
docker-compose logs -f streamlit-app
```

---

## 📈 Tamanho da Imagem

### **Estimativa de Tamanho**
- **Base Image** (`python:3.11-slim`): ~150 MB
- **Dependências Python**: ~300-400 MB
- **Código + Modelo**: ~10-20 MB
- **Total Aproximado**: **~460-570 MB**

### **Verificar Tamanho Real**
```bash
docker images creditcard-risk:latest
```

---

## 🚀 Deploy em Produção

### **Opção 1: Docker Swarm**
```bash
docker stack deploy -c docker-compose.yml creditrisk
```

### **Opção 2: Kubernetes**
```bash
# Gerar manifests do Kompose
kompose convert -f docker-compose.yml
kubectl apply -f .
```

### **Opção 3: Cloud (AWS ECS, Azure Container Instances, GCP Cloud Run)**
```bash
# Push para registry
docker tag creditcard-risk:latest <registry>/creditcard-risk:latest
docker push <registry>/creditcard-risk:latest
```

---

## 📝 Boas Práticas

✅ **Sempre treinar o modelo antes do build**
```bash
cd src && python train_model.py
```

✅ **Usar tags versionadas em produção**
```bash
docker build -t creditcard-risk:1.0.0 .
```

✅ **Fazer backup do modelo**
```bash
docker cp creditcard-risk-app:/app/models/modelo_credito.pkl ./backup/
```

✅ **Monitorar logs regularmente**
```bash
docker-compose logs -f --tail=50
```

---

## 🤝 Contribuindo

Para melhorias no Dockerfile ou docker-compose.yml, consulte as [contribuição guidelines](../README.md).
