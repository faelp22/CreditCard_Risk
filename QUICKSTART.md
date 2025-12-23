# 🚀 Como Executar o Projeto

## � Opção 1: Docker (Recomendado para Produção)

A forma mais rápida e segura de executar o projeto:

### Windows (PowerShell)
```powershell
# Build e start
.\docker-manage.ps1 build
.\docker-manage.ps1 start

# Ver logs
.\docker-manage.ps1 logs

# Parar
.\docker-manage.ps1 stop
```

### Linux/Mac (Bash)
```bash
# Dar permissão de execução
chmod +x docker-manage.sh

# Build e start
./docker-manage.sh build
./docker-manage.sh start

# Ver logs
./docker-manage.sh logs

# Parar
./docker-manage.sh stop
```

### Comandos Docker Diretos
```bash
# Build
docker-compose build

# Start (modo background)
docker-compose up -d

# Ver logs
docker-compose logs -f

# Parar
docker-compose down
```

**Acesse:** http://localhost:8501

📖 **[Documentação Completa do Docker](docs/DOCKER.md)**

---

## 🖥️ Opção 2: Instalação Local

## �📂 Estrutura Organizada

Agora o projeto está organizado em diretórios:

```
CreditCard_Risk/
├── 📁 src/                    # Código-fonte
│   ├── app.py                 # Aplicação Streamlit
│   ├── train_model.py         # Script de treinamento
│   ├── llm.py                 # Integração com LLM
│   ├── explain.py             # Módulo de explicabilidade (SHAP)
│   └── prompts.py             # Templates de prompts
│
├── 📁 data/                   # Datasets
│   └── UCI_Credit_Card.csv    # Dataset original
│
├── 📁 models/                 # Modelos treinados
│   └── modelo_credito.pkl     # Pipeline ML treinado
│
├── 📁 reports/                # Relatórios e visualizações
│   ├── MODEL_REPORT.md        # Relatório técnico detalhado
│   ├── confusion_matrix.png   # Matriz de confusão
│   └── roc_curve.png          # Curva ROC
│
├── 📁 notebooks/              # Jupyter Notebooks
│   └── credit_EDA.ipynb       # Análise exploratória
│
├── 📁 docs/                   # Documentação
│   └── SETUP.md               # Guia de instalação detalhado
│
├── 📄 requirements.txt        # Dependências (produção)
├── 📄 requirements-jupyter.txt # Dependências (desenvolvimento)
├── 📄 .gitignore              # Arquivos ignorados pelo Git
└── 📄 README.md               # Este arquivo
```

---

## ⚡ Quick Start (Local)

### 1️⃣ Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2️⃣ Treinar o Modelo
```bash
cd src
python train_model.py
```

### 3️⃣ Executar Aplicação
```bash
cd src
streamlit run app.py
```

## 📖 Documentação Completa

- 📝 **[Guia de Instalação Detalhado](docs/SETUP.md)**
- 📊 **[Relatório do Modelo](reports/MODEL_REPORT.md)**
- 📓 **[Análise Exploratória](notebooks/credit_EDA.ipynb)**

## 🎯 Principais Comandos

```bash
# Treinar/Retreinar modelo
cd src && python train_model.py

# Executar aplicação Streamlit
cd src && streamlit run app.py

# Abrir notebook de análise
jupyter notebook notebooks/credit_EDA.ipynb
```

## 📊 O Projeto

Sistema completo de análise de risco de crédito combinando:
- 🤖 Machine Learning (Random Forest)
- 🧠 IA Generativa (LLM para explicações)
- 📊 Visualizações interativas (Streamlit)
- 🔍 Explicabilidade (SHAP values)
