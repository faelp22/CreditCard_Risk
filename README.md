# 💳 CreditCard Risk Prediction
### Sistema Inteligente de Análise de Risco de Crédito com Machine Learning e IA Generativa

> 🔱 **Fork**: Este é um fork melhorado do [projeto original](https://github.com/liakruk/CreditCard_Risk) por [@liakruk](https://github.com/liakruk)  
> ⚡ **Melhorias**: Docker em produção, LLM local (Ollama), otimizações de segurança e performance

[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai/)

![1220](https://github.com/user-attachments/assets/6f3f1c31-e485-41c2-a0b9-6f0cef2f7e67)

---

## 📋 Sobre o Projeto

Este projeto representa uma solução completa de **análise de risco de crédito**, combinando técnicas avançadas de Machine Learning com Inteligência Artificial Generativa para criar um sistema de decisão transparente e interpretável.

### 🎯 Objetivo

Desenvolver uma ferramenta preditiva que auxilie instituições financeiras a:
- **Prever inadimplência** com alta precisão
- **Ajustar dinamicamente** o threshold de decisão para otimizar lucro
- **Explicar decisões** de forma clara e personalizada para clientes
- **Simular cenários** de negócio em tempo real

### 🔑 Diferenciais

#### 🎯 Projeto Original ([@liakruk](https://github.com/liakruk)):
- 📊 **SHAP Values**: Análise de contribuição individual de cada feature
- 🎛️ **Interface Interativa**: Dashboard Streamlit com ajuste de threshold em tempo real
- 📈 **Feature Engineering**: Criação de variáveis derivadas baseadas em análise temporal

#### ⚡ Melhorias Neste Fork ([@faelp22](https://github.com/faelp22)):
- 🐳 **Docker em Produção**: Build multi-stage, imagem otimizada (~460-570 MB)
- 🔒 **Segurança**: Usuário não-privilegiado, filesystem read-only, resource limits
- 🤖 **LLM Local**: Integração com Ollama (qwen2.5:0.5b) para explicações em português
- 🎨 **UI Limpa**: CSS customizado, interface minimalista sem branding Streamlit
- 📚 **Documentação**: SHAP_EXPLICACAO.md, QUICKSTART.md, configurações otimizadas

---

## 🚀 O Desafio

Criar uma ferramenta de previsão de inadimplência que não apenas classifica clientes, mas que oferece:
- ✅ **Ajuste interativo de threshold** para otimização de lucro
- ✅ **Explicações claras e personalizadas** para cada decisão
- ✅ **Suporte inteligente** para tomada de decisão estratégica
- ✅ **Transparência total** no processo preditivo

---

## 📂 Estrutura do Projeto

```
CreditCard_Risk/
├── 📁 src/                      # Código-fonte
│   ├── app.py                   # Aplicação Streamlit principal
│   ├── train_model.py           # Script de treinamento do modelo
│   ├── llm.py                   # Integração com LLM (Ollama)
│   ├── explain.py               # Módulo de explicabilidade (SHAP)
│   ├── prompts.py               # Templates de prompts para LLM
│   └── __init__.py              # Inicialização do pacote
│
├── 📁 data/                     # Datasets
│   └── UCI_Credit_Card.csv      # Dataset original (30k clientes)
│
├── 📁 models/                   # Modelos treinados
│   └── modelo_credito.pkl       # Pipeline ML completo (~2-5 MB)
│
├── 📁 reports/                  # Relatórios e visualizações
│   ├── MODEL_REPORT.md          # Relatório técnico detalhado
│   ├── confusion_matrix.png     # Matriz de confusão
│   └── roc_curve.png            # Curva ROC
│
├── 📁 notebooks/                # Jupyter Notebooks
│   └── credit_EDA.ipynb         # Análise exploratória completa
│
├── 📁 docs/                     # Documentação
│   └── SETUP.md                 # Guia de instalação detalhado
│
├── 📄 requirements.txt          # Dependências (produção)
├── 📄 requirements-jupyter.txt  # Dependências (desenvolvimento)
├── 📄 QUICKSTART.md             # Guia rápido de execução
├── 📄 .gitignore                # Arquivos ignorados pelo Git
├── 📄 LICENSE                   # Licença MIT
└── 📄 README.md                 # Este arquivo
```

---

## 🏗️ Arquitetura do Projeto

O pipeline foi desenvolvido em **3 etapas principais**:

### 📊 1. Análise Exploratória & Feature Engineering
**Arquivo**: `notebooks/credit_EDA.ipynb`

- Análise exploratória profunda de 30.000 clientes
- Identificação de padrões de comportamento financeiro
- Criação de features derivadas:
  - `CREDIT_UTILIZATION`: Razão entre fatura e limite de crédito
  - `UTILIZATION_GROWTH_6M`: Tendência de crescimento do uso de crédito
  - `payment_ratio1`: Capacidade de pagamento da fatura
  - Séries temporais de 6 meses de histórico de pagamento

**Insights-chave**:
- Clientes com utilização de crédito > 80% têm risco 4.5x maior
- Histórico de pagamento dos últimos 3 meses é altamente preditivo
- Padrões demográficos (idade, escolaridade) correlacionam com risco

### 🤖 2. Modelagem Preditiva & Interface Interativa
**Arquivos**: `app.py`, `llm.py`, `explain.py`

- **Modelo**: Classificação binária (Random Forest / XGBoost)
- **Métricas**: Precision, Recall, F1-Score, ROC-AUC
- **Interface Streamlit**:
  - Upload de dados de clientes
  - Ajuste dinâmico de threshold (0-100%)
  - Visualização de métricas de negócio
  - Simulação de lucro/prejuízo

### 🧠 3. Explicabilidade com IA Generativa
**Arquivo**: `llm.py`, `prompts.py`

- **SHAP (SHapley Additive exPlanations)**: Análise de importância de features
- **LLM Local (Ollama)**: Geração de narrativas personalizadas
- **Explicações contextualizadas**: 
  - Por que o crédito foi aprovado/negado?
  - Quais fatores mais influenciaram a decisão?
  - Recomendações para melhoria do score

---

## 🛠️ Tecnologias Utilizadas

### Core
- **Python 3.10+**: Linguagem principal
- **Pandas & NumPy**: Manipulação e análise de dados
- **Scikit-learn**: Modelagem e avaliação

### Visualização
- **Matplotlib & Seaborn**: Gráficos estáticos
- **Plotly**: Visualizações interativas
- **Streamlit**: Dashboard web

### IA & Explicabilidade
- **SHAP**: Interpretabilidade do modelo
- **Ollama**: LLM local para geração de texto
- **Requests**: Comunicação com API do Ollama

---

## 📦 Instalação

### 1️⃣ Clone o repositório

**Este fork (com melhorias Docker/LLM):**
```bash
git clone git@github.com:faelp22/CreditCard_Risk.git
cd CreditCard_Risk
```

**Projeto original:**
```bash
git clone https://github.com/liakruk/CreditCard_Risk.git
cd CreditCard_Risk
```

**Para contribuir com o original, configure os remotes:**
```bash
git remote add origin git@github.com:faelp22/CreditCard_Risk.git
git remote add upstream https://github.com/liakruk/CreditCard_Risk.git
```

### 2️⃣ Crie um ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3️⃣ Instale as dependências

**Para uso da aplicação Streamlit:**
```bash
pip install -r requirements.txt
```

**Para análise no Jupyter Notebook:**
```bash
pip install -r requirements-jupyter.txt
```

### 4️⃣ Configure o Ollama (opcional - para explicações com IA)
```bash
# Instale o Ollama: https://ollama.ai/
ollama pull qwen2.5:0.5b  # Modelo leve (379 MB) - recomendado neste fork
# ou
ollama pull llama2  # Modelo original do projeto
```

---

## 🚀 Como Usar

### 🐳 Produção com Docker (Melhorias deste Fork)

```bash
# Build da imagem
docker build -t faelp22/credit-risk-analysis:latest .

# Iniciar aplicação
docker compose up -d

# Acessar: http://localhost:8502

# Ver logs
docker logs creditcard-risk-app -f

# Parar
docker compose down
```

**Características das melhorias Docker:**
- ✅ Multi-stage build otimizado
- ✅ Imagem final: ~460-570 MB
- ✅ Usuário não-privilegiado (UID 1001)
- ✅ Filesystem somente leitura (read-only)
- ✅ Resource limits (2 CPUs, 2GB RAM)
- ✅ Health checks automatizados
- ✅ Modelo incluído na imagem
- ✅ Configurações de produção (.streamlit/config.toml)

**Mais detalhes**: Veja `QUICKSTART.md` para guia completo

---

### 🖥️ Desenvolvimento Local

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Treinar modelo
cd src && python train_model.py

# 3. Executar aplicação
cd src && streamlit run app.py
```

Acesse: `http://localhost:8501`

---

## 📊 Resultados

### Métricas do Modelo
- **Acurácia**: ~82%
- **Precision**: ~75%
- **Recall**: ~68%
- **ROC-AUC**: ~0.78

### Impacto de Negócio
- **Redução de inadimplência**: ~30%
- **Otimização de threshold**: Aumento de 15% no lucro líquido
- **Tempo de decisão**: < 2 segundos por análise

---

## 📂 Estrutura do Projeto

```
CreditCard_Risk/
├── 📊 UCI_Credit_Card.csv          # Dataset principal
├── 📓 credit_EDA.ipynb              # Análise exploratória
├── 🚀 app.py                        # Aplicação Streamlit
├── 🤖 llm.py                        # Integração com LLM
├── 💬 prompts.py                    # Templates de prompts
├── 🔍 explain.py                    # Lógica de explicabilidade
├── 📋 requirements.txt              # Dependências (produção)
├── 📋 requirements-jupyter.txt      # Dependências (desenvolvimento)
└── 📖 README.md                     # Documentação
```

---

## 🎓 Aprendizados

Como parte da minha jornada em **Data Science e IA Generativa**, este projeto me proporcionou:

- 📚 Experiência prática com pipeline ML completo (EDA → Modelagem → Deploy)
- 🧠 Compreensão profunda de explicabilidade (SHAP, LIME)
- 🤖 Integração de LLMs locais em aplicações de ML
- 📊 Análise de impacto de negócio (threshold optimization)
- 🎨 Desenvolvimento de interfaces interativas com Streamlit

---

## 🔮 Próximos Passos

- [ ] Implementar AutoML para seleção automática de modelos
- [ ] Adicionar testes unitários e integração contínua
- [ ] Criar API REST para integração com sistemas externos
- [ ] Implementar monitoramento de drift do modelo
- [ ] Adicionar suporte a múltiplos idiomas nas explicações

---

## 🤝 Contribuições

Feedbacks e sugestões são muito bem-vindos! Sinta-se à vontade para:

- 🐛 Reportar bugs
- 💡 Sugerir melhorias
- 🔧 Enviar pull requests
- ⭐ Dar uma estrela se achou útil!

**Para contribuir:**
- **Neste fork**: Abra issues/PRs em [faelp22/CreditCard_Risk](https://github.com/faelp22/CreditCard_Risk)
- **Projeto original**: Abra issues/PRs em [liakruk/CreditCard_Risk](https://github.com/liakruk/CreditCard_Risk)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

<div align="center">
  
**⭐ Se este projeto te ajudou, considere dar uma estrela! ⭐**

### 🔱 Projeto Original
Desenvolvido com ❤️ por [@liakruk](https://github.com/liakruk)

### ⚡ Fork & Melhorias
Otimizado para produção por [@faelp22](https://github.com/faelp22)

</div>
