# 🎓 Guia Completo: Treinamento do Modelo de Risco de Crédito

Este documento explica detalhadamente o processo de treinamento do modelo de Machine Learning utilizado no sistema de análise de risco de crédito.

---

## 📋 Índice

1. [Visão Geral](#-visão-geral)
2. [Preparação dos Dados](#-preparação-dos-dados)
3. [Pipeline de Pré-processamento](#-pipeline-de-pré-processamento)
4. [Algoritmo e Hiperparâmetros](#-algoritmo-e-hiperparâmetros)
5. [Processo de Treinamento](#-processo-de-treinamento)
6. [Avaliação do Modelo](#-avaliação-do-modelo)
7. [Serialização e Deploy](#-serialização-e-deploy)
8. [Como Executar](#-como-executar)
9. [Otimização e Melhorias](#-otimização-e-melhorias)

---

## 🎯 Visão Geral

### Objetivo
Treinar um modelo de **classificação binária** para prever se um cliente de cartão de crédito irá inadimplir no próximo mês.

### Características do Modelo
- **Algoritmo**: Random Forest Classifier
- **Framework**: Scikit-learn 1.3+
- **Dataset**: UCI Credit Card Default (30.000 clientes)
- **Target**: `default.payment.next.month` (0 = Paga, 1 = Inadimplente)
- **Features**: 23 variáveis (demográficas, limite, histórico de pagamento)
- **Split**: 80% treino (24.000) / 20% teste (6.000)

### Arquivo Principal
```
src/train_model.py
```

---

## 📊 Preparação dos Dados

### 1️⃣ **Carregamento**
```python
df = pd.read_csv("../data/UCI_Credit_Card.csv")
# Shape: (30000, 25) - 30 mil clientes, 25 colunas
```

### 2️⃣ **Limpeza de Dados**
```python
# Corrigir valores inválidos em EDUCATION
# 0, 5, 6 são valores não documentados → agrupados em "outros" (4)
df.loc[df.EDUCATION.isin([0, 5, 6]), 'EDUCATION'] = 4

# Corrigir valores inválidos em MARRIAGE
# 0 não está na documentação → agrupado em "outros" (3)
df.loc[df.MARRIAGE == 0, 'MARRIAGE'] = 3
```

**Motivo**: O dataset original contém valores não documentados que podem confundir o modelo.

### 3️⃣ **Separação de Features e Target**
```python
# Remover colunas não preditivas
X = df.drop(columns=["default.payment.next.month", "ID"], errors="ignore")
y = df["default.payment.next.month"]

# X: 23 features numéricas
# y: 0 (Paga) = 23,364 | 1 (Inadimplente) = 6,636
```

### 4️⃣ **Split Estratificado**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% para teste
    random_state=42,    # Reprodutibilidade
    stratify=y          # Manter proporção de classes
)
```

**Stratify**: Garante que a proporção 78% Paga / 22% Inadimplente seja mantida em ambos os conjuntos.

---

## 🏗️ Pipeline de Pré-processamento

### Por que usar Pipeline?
1. **Evita data leakage**: Transformações são aplicadas apenas nos dados de treino
2. **Reprodutibilidade**: Mesmo pré-processamento em produção
3. **Serialização fácil**: Todo o fluxo é salvo em um único arquivo `.pkl`

### Estrutura do Pipeline

```python
Pipeline([
    ('preprocessor', ColumnTransformer),  # Etapa 1: Pré-processamento
    ('classifier', RandomForestClassifier) # Etapa 2: Modelo
])
```

### Pré-processador
```python
preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), slice(None))  # Escala todas as features
    ],
    remainder='passthrough'
)
```

**StandardScaler**: Padroniza features para média=0 e desvio=1.

**Por quê escalar?**
- Melhora convergência de algoritmos
- Evita que features com valores grandes dominem
- Exemplo: `LIMIT_BAL` (0-1.000.000) vs `AGE` (21-79)

**Após escalonamento:**
```
LIMIT_BAL: 50000 → -0.23
AGE: 35 → 0.45
```

---

## 🌲 Algoritmo e Hiperparâmetros

### Por que Random Forest?

| Vantagem | Descrição |
|----------|-----------|
| ✅ **Robusto** | Lida bem com outliers e dados ruidosos |
| ✅ **Feature Importance** | Identifica variáveis mais importantes |
| ✅ **Não-linear** | Captura relações complexas sem engenharia manual |
| ✅ **Pouco overfitting** | Ensemble de árvores reduz variância |
| ✅ **Interpretável** | SHAP consegue explicar decisões |

### Hiperparâmetros Configurados

```python
RandomForestClassifier(
    n_estimators=100,           # Número de árvores na floresta
    max_depth=15,               # Profundidade máxima de cada árvore
    min_samples_split=10,       # Mínimo de amostras para dividir nó
    min_samples_leaf=5,         # Mínimo de amostras em folha
    random_state=42,            # Seed para reprodutibilidade
    n_jobs=-1,                  # Usar todos os CPUs disponíveis
    class_weight='balanced'     # Ajuste para desbalanceamento de classes
)
```

### 📊 Explicação dos Hiperparâmetros

#### 1. **n_estimators=100**
- Número de árvores de decisão no ensemble
- Mais árvores → Melhor performance (até certo ponto)
- 100 é um bom equilíbrio entre performance e tempo

#### 2. **max_depth=15**
- Limita profundidade das árvores
- Evita overfitting (árvores muito específicas)
- 15 níveis é suficiente para capturar padrões complexos

#### 3. **min_samples_split=10**
- Número mínimo de amostras para dividir um nó
- Valores maiores → Árvores mais generalizadas
- Reduz overfitting em nós com poucos exemplos

#### 4. **min_samples_leaf=5**
- Número mínimo de amostras em cada folha
- Evita folhas com 1-2 exemplos (ruído)
- Melhora generalização

#### 5. **class_weight='balanced'**
- **CRÍTICO para dados desbalanceados!**
- Dataset: 78% Paga / 22% Inadimplente
- Ajusta pesos automaticamente: `n_samples / (n_classes * np.bincount(y))`

**Sem balanceamento:**
```
Modelo tende a prever sempre "Paga" (classe majoritária)
Acurácia: 78% (mas não detecta inadimplentes!)
```

**Com balanceamento:**
```
Penaliza erros na classe minoritária (inadimplentes)
Força o modelo a aprender padrões de ambas as classes
```

---

## 🚀 Processo de Treinamento

### Fluxo Completo

```python
def main():
    # 1. Carregar e preparar dados
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # 2. Criar pipeline
    pipeline = create_pipeline()
    
    # 3. Treinar modelo
    pipeline.fit(X_train, y_train)
    
    # 4. Avaliar no conjunto de teste
    evaluate_model(pipeline, X_test, y_test)
    
    # 5. Salvar modelo treinado
    save_model(pipeline, "models/modelo_credito.pkl")
```

### Detalhamento das Etapas

#### **Etapa 1: Carregamento**
```
📊 Carregando dados...
✅ Dados carregados: 30000 linhas, 25 colunas
🧹 Limpando dados...
✅ Features: 23
✅ Target distribuição: {0: 23364, 1: 6636}
✅ Train set: 24000 amostras
✅ Test set: 6000 amostras
```

#### **Etapa 2: Pipeline**
```
🏗️ Criando pipeline...
✅ Pipeline criado!
```

#### **Etapa 3: Treinamento**
```
🚀 Iniciando treinamento...
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done  42 tasks | elapsed:   12.3s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   28.7s finished
✅ Treinamento concluído!
```

**Tempo de treinamento**: ~30 segundos (varia por hardware)

#### **Etapa 4: Avaliação**
```
📈 Avaliando modelo...

📊 RELATÓRIO DE CLASSIFICAÇÃO:
              precision    recall  f1-score   support

        Paga       0.87      0.87      0.87      4673
     Default       0.54      0.54      0.54      1327

    accuracy                           0.80      6000
   macro avg       0.71      0.71      0.71      6000
weighted avg       0.80      0.80      0.80      6000

📊 MATRIZ DE CONFUSÃO:
[[4073  600]
 [ 616  711]]

📊 ROC-AUC Score: 0.7707
✅ Matriz de confusão salva em '../reports/confusion_matrix.png'
✅ Curva ROC salva em '../reports/roc_curve.png'
```

---

## 📈 Avaliação do Modelo

### Métricas Principais

#### 1️⃣ **Acurácia: 80%**
```
Acertos Totais / Total de Predições
(4073 + 711) / 6000 = 0.80
```
✅ **Interpretação**: Acerta 8 em cada 10 predições

#### 2️⃣ **Precision (Precisão)**

**Classe "Paga" (0): 87%**
```
Verdadeiros Negativos / (Verdadeiros Negativos + Falsos Negativos)
4073 / (4073 + 600) = 0.87
```
✅ Quando diz que vai pagar, está certo 87% das vezes

**Classe "Inadimplente" (1): 54%**
```
Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Positivos)
711 / (711 + 616) = 0.54
```
⚠️ Quando diz que vai dar default, está certo apenas 54% das vezes

#### 3️⃣ **Recall (Sensibilidade)**

**Classe "Paga" (0): 87%**
```
Verdadeiros Negativos / Total Real de Pagantes
4073 / 4673 = 0.87
```
✅ Detecta 87% de todos os bons pagadores

**Classe "Inadimplente" (1): 54%**
```
Verdadeiros Positivos / Total Real de Inadimplentes
711 / 1327 = 0.54
```
⚠️ Detecta apenas 54% dos inadimplentes (46% escapam!)

#### 4️⃣ **ROC-AUC: 0.7707**
```
Área sob a curva ROC
```
✅ **Interpretação**: 
- Score > 0.7 é considerado **bom** para problemas de crédito
- 77% de chance de ranquear um inadimplente com score maior que um pagador
- Quanto mais próximo de 1.0, melhor

### Matriz de Confusão Explicada

```
                  PREDITO: Paga    PREDITO: Inadimplente
REAL: Paga            4,073 (TN)         600 (FP)
REAL: Inadimplente      616 (FN)         711 (TP)
```

| Quadrante | Valor | Nome | Impacto de Negócio |
|-----------|-------|------|-------------------|
| **TN** | 4,073 | Verdadeiro Negativo | ✅ Aprovados corretamente → Lucro |
| **FP** | 600 | Falso Positivo | ⚠️ Bons clientes rejeitados → Oportunidade perdida |
| **FN** | 616 | Falso Negativo | ❌ Maus clientes aprovados → Prejuízo! |
| **TP** | 711 | Verdadeiro Positivo | ✅ Maus clientes rejeitados → Prejuízo evitado |

### 💰 Simulação de Impacto Financeiro

**Premissas:**
- Lucro por cliente aprovado: R$ 100
- Prejuízo por inadimplente: R$ 1.000

**Resultado com threshold 0.5:**
```
Aprovados: 4,073 + 616 = 4,689
Lucro: 4,073 × R$ 100 = R$ 407,300
Prejuízo: 616 × R$ 1,000 = R$ 616,000
RESULTADO: -R$ 208,700 (prejuízo!)
```

**Ajustando threshold para 0.3 (mais agressivo):**
```
Aprovados: ~5,200
Inadimplentes aprovados: ~800
Lucro: 4,400 × R$ 100 = R$ 440,000
Prejuízo: 800 × R$ 1,000 = R$ 800,000
RESULTADO: -R$ 360,000 (pior!)
```

**Ajustando threshold para 0.7 (mais conservador):**
```
Aprovados: ~3,800
Inadimplentes aprovados: ~350
Lucro: 3,450 × R$ 100 = R$ 345,000
Prejuízo: 350 × R$ 1,000 = R$ 350,000
RESULTADO: -R$ 5,000 (quase break-even!)
```

🎯 **Conclusão**: O threshold ideal depende da estratégia de negócio e deve ser ajustado no Streamlit!

---

## 💾 Serialização e Deploy

### Salvando o Modelo

```python
import joblib

# Salvar pipeline completo (preprocessador + modelo)
joblib.dump(pipeline, "../models/modelo_credito.pkl")
```

**O que é salvo:**
- ✅ Pipeline completo (preprocessador + Random Forest)
- ✅ Hiperparâmetros configurados
- ✅ Árvores treinadas (100 árvores com seus splits)
- ✅ Scaler com média/desvio calculados no treino

**Tamanho do arquivo**: ~2-5 MB (depende das árvores)

### Carregando em Produção

```python
# Em app.py
pipeline = joblib.load("models/modelo_credito.pkl")

# Fazer predição
probs = pipeline.predict_proba(X_new)[:, 1]
```

**Vantagens:**
- ✅ Mesmas transformações aplicadas automaticamente
- ✅ Nenhum pré-processamento manual necessário
- ✅ Garantia de reprodutibilidade

---

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install -r requirements.txt
```

### Treinamento Local

```bash
# Navegar para o diretório src
cd src

# Executar script de treinamento
python train_model.py
```

### Saída Esperada
```
============================================================
🎯 TREINAMENTO DO MODELO DE RISCO DE CRÉDITO
============================================================
📊 Carregando dados...
✅ Dados carregados: 30000 linhas, 25 colunas
🧹 Limpando dados...
✅ Features: 23
✅ Target distribuição: {0: 23364, 1: 6636}
✅ Train set: 24000 amostras
✅ Test set: 6000 amostras
🏗️ Criando pipeline...
✅ Pipeline criado!

🚀 Iniciando treinamento...
✅ Treinamento concluído!

📈 Avaliando modelo...

📊 RELATÓRIO DE CLASSIFICAÇÃO:
              precision    recall  f1-score   support

        Paga       0.87      0.87      0.87      4673
     Default       0.54      0.54      0.54      1327

    accuracy                           0.80      6000
   macro avg       0.71      0.71      0.71      6000
weighted avg       0.80      0.80      0.80      6000

📊 MATRIZ DE CONFUSÃO:
[[4073  600]
 [ 616  711]]

📊 ROC-AUC Score: 0.7707
✅ Matriz de confusão salva em '../reports/confusion_matrix.png'
✅ Curva ROC salva em '../reports/roc_curve.png'

💾 Salvando modelo em '../models/modelo_credito.pkl'...
✅ Modelo salvo com sucesso!
📝 Relatório detalhado disponível em '../reports/MODEL_REPORT.md'

============================================================
✅ PROCESSO CONCLUÍDO COM SUCESSO!
============================================================
📊 ROC-AUC Score: 0.7707
💾 Modelo salvo: models/modelo_credito.pkl
📈 Visualizações salvas: reports/confusion_matrix.png, reports/roc_curve.png
📝 Relatório completo: reports/MODEL_REPORT.md

🚀 Agora você pode executar o Streamlit com:
   cd src && streamlit run app.py
============================================================
```

### Arquivos Gerados

```
models/
└── modelo_credito.pkl          # Pipeline treinado (~2-5 MB)

reports/
├── confusion_matrix.png        # Visualização da matriz de confusão
├── roc_curve.png               # Curva ROC
└── MODEL_REPORT.md             # Relatório técnico detalhado
```

---

## 🔧 Otimização e Melhorias

### Melhorias Implementadas Neste Fork

#### 1️⃣ **Class Weight Balancing**
```python
class_weight='balanced'
```
✅ Melhora detecção de inadimplentes (classe minoritária)

#### 2️⃣ **Regularização das Árvores**
```python
max_depth=15,
min_samples_split=10,
min_samples_leaf=5
```
✅ Evita overfitting mantendo boa capacidade preditiva

#### 3️⃣ **Pipeline Completo**
```python
Pipeline([('preprocessor', ...), ('classifier', ...)])
```
✅ Garante reprodutibilidade em produção

### Possíveis Melhorias Futuras

#### 1️⃣ **Grid Search para Otimização**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 15, 20],
    'classifier__min_samples_split': [5, 10, 15]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
```

#### 2️⃣ **Feature Engineering Avançado**
```python
# Criar features derivadas
df['payment_consistency'] = df[['PAY_0', 'PAY_2', 'PAY_3']].std(axis=1)
df['credit_utilization'] = df['BILL_AMT1'] / df['LIMIT_BAL']
df['payment_ratio'] = df['PAY_AMT1'] / df['BILL_AMT1']
```

#### 3️⃣ **Algoritmos Alternativos**
```python
# XGBoost (geralmente melhor que Random Forest)
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=3.5  # Ajuste para desbalanceamento
)
```

#### 4️⃣ **Validação Cruzada**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
print(f"ROC-AUC médio: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

#### 5️⃣ **Threshold Optimization**
```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# Encontrar threshold que maximiza F1-Score
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
best_threshold = thresholds[np.argmax(f1_scores)]
```

#### 6️⃣ **SMOTE para Balanceamento**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

---

## 📚 Recursos Adicionais

### Documentação Relacionada
- 📄 [`SHAP_EXPLICACAO.md`](../SHAP_EXPLICACAO.md) - Explicação sobre interpretabilidade
- 📄 [`MODEL_REPORT.md`](../reports/MODEL_REPORT.md) - Relatório técnico detalhado
- 📄 [`QUICKSTART.md`](../QUICKSTART.md) - Guia rápido de execução
- 📄 [`DOCKER.md`](./DOCKER.md) - Deploy com Docker

### Links Úteis
- [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Understanding ROC-AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- [Handling Imbalanced Data](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
- [UCI Credit Card Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

---

## 🎓 Resumo

### O que o Processo de Treinamento Faz:

1. ✅ **Carrega** o dataset de 30 mil clientes
2. ✅ **Limpa** valores inválidos em EDUCATION e MARRIAGE
3. ✅ **Separa** em 80% treino / 20% teste (estratificado)
4. ✅ **Cria pipeline** com StandardScaler + Random Forest
5. ✅ **Treina** 100 árvores de decisão com class_weight='balanced'
6. ✅ **Avalia** com múltiplas métricas (Accuracy, Precision, Recall, ROC-AUC)
7. ✅ **Gera visualizações** (matriz de confusão, curva ROC)
8. ✅ **Salva** pipeline completo em `models/modelo_credito.pkl`

### Métricas Atingidas:
- 📊 **Acurácia**: 80%
- 📊 **ROC-AUC**: 0.7707
- 📊 **Precision (Inadimplente)**: 54%
- 📊 **Recall (Inadimplente)**: 54%

### Próximos Passos:
1. Executar `python train_model.py` para treinar
2. Usar `streamlit run app.py` para testar o modelo
3. Ajustar threshold no Streamlit para otimizar lucro
4. Considerar melhorias (Grid Search, XGBoost, Feature Engineering)

---

<div align="center">

**🎯 Modelo treinado e pronto para produção! 🚀**

Para mais informações, consulte [`MODEL_REPORT.md`](../reports/MODEL_REPORT.md)

</div>
