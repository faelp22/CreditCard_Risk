# SHAP: Explicando Decisões de Machine Learning

**SHAP** (SHapley Additive exPlanations) é uma técnica de **explicabilidade de IA** baseada na teoria dos jogos que responde à pergunta: **"Por que o modelo tomou essa decisão?"**

## 🎯 O Problema que SHAP Resolve

Modelos de ML como Random Forest são "caixas-pretas" — eles fazem previsões precisas, mas não dizem **por quê**. No seu caso:

```
Cliente X → [Random Forest] → 85% de risco ❌ Reprovado
                    ↑
              Por quê 85%? 🤔
```

## 🔍 Como SHAP Funciona

SHAP atribui um **valor de contribuição** para cada feature, mostrando o quanto ela **aumentou** ou **diminuiu** a probabilidade de risco:

### Exemplo Real do Sistema:
```python
Cliente com 85% de risco de inadimplência:

Feature                    | Valor SHAP | Interpretação
---------------------------|------------|--------------------------------
PAY_0 (atraso atual)       | +0.32      | Atraso de 2 meses → aumenta muito o risco
LIMIT_BAL (limite)         | -0.08      | Limite alto de R$ 50k → reduz o risco
PAY_AMT1 (pagamento)       | -0.05      | Pagou R$ 2k no último mês → reduz risco
AGE (idade)                | +0.02      | 23 anos → aumenta levemente o risco
```

## 💡 Por Que SHAP é Usado Neste Sistema?

### 1️⃣ **Conformidade Legal (Lei Geral de Proteção de Dados)**
```
❌ "Crédito negado"  
✅ "Crédito negado porque você tem 2 meses de atraso 
    e histórico de pagamento irregular"
```

### 2️⃣ **Geração de Prompts para o LLM**
No código (`explain.py`), SHAP é usado para alimentar o Ollama:

```python
# 1. SHAP extrai os fatores
shap_values = compute_shap_single(pipeline, cliente)
fatores = extract_shap_factors(shap_values, top_k=5)

# 2. Fatores vão para o prompt do LLM
prompt = build_credit_prompt(
    decision="Reprovado",
    prob=0.85,
    factors=fatores  # ← AQUI entra o SHAP!
)

# 3. LLM gera explicação humanizada
"Seu crédito foi negado principalmente devido ao atraso 
de 2 meses no pagamento. Recomendamos regularizar..."
```

### 3️⃣ **Precisão vs. Genérico**
```
Sem SHAP (LLM sozinho):
"Seu crédito foi negado por questões de histórico financeiro"
                    ↑ Vago e inútil

Com SHAP + LLM:
"Os 2 fatores principais foram:
 1. Atraso de 2 meses (PAY_0) - impacto alto
 2. Limite baixo de R$ 5k - impacto médio
 Sugestão: Regularize os pagamentos em atraso..."
                    ↑ Específico e acionável
```

## 🛠️ Como Funciona no Pipeline

```
Cliente → Random Forest → Probabilidade 85%
                              ↓
                         [SHAP Explainer]
                              ↓
                    Top 5 features + valores
                              ↓
                      [Prompt Engineering]
                              ↓
                          Ollama LLM
                              ↓
                   Explicação em português
```

### Fluxo Detalhado:

1. **Random Forest** gera probabilidade de risco (ex: 85%)
2. **SHAP Explainer** calcula contribuição de cada feature
3. **Top K Features** são extraídas (5 mais relevantes)
4. **Prompt Builder** monta contexto estruturado para LLM
5. **Ollama (qwen2.5:0.5b)** gera explicação em linguagem natural
6. **Cliente** recebe explicação personalizada e acionável

## 📊 Visualizando SHAP

O sistema gera uma tabela com os principais fatores:

| Feature | Valor | Impacto SHAP | Direção |
|---------|-------|--------------|---------|
| PAY_0   | 2 meses | +0.32 | 🔴 Aumenta risco |
| LIMIT_BAL | R$ 50k | -0.08 | 🟢 Reduz risco |
| PAY_AMT1 | R$ 2k | -0.05 | 🟢 Reduz risco |
| BILL_AMT1 | R$ 15k | +0.03 | 🔴 Aumenta risco |
| PAY_2 | 1 mês | +0.02 | 🔴 Aumenta risco |

## 🎓 Teoria: Shapley Values (Nobel de Economia 2012)

SHAP usa a teoria dos **Shapley Values**, que responde:

> "Se cada feature fosse um 'jogador' contribuindo para a decisão, qual seria a contribuição justa de cada uma?"

É como dividir um prêmio de equipe considerando a contribuição individual de cada membro.

### Propriedades Matemáticas:

1. **Aditividade**: Soma dos valores SHAP = diferença entre predição e valor base
2. **Consistência**: Se uma feature contribui mais, seu valor SHAP é maior
3. **Simetria**: Features com mesma contribuição têm mesmo valor SHAP
4. **Dummy**: Features irrelevantes têm valor SHAP = 0

## ✅ Vantagens no Caso de Uso de Crédito

1. **Confiança**: Clientes entendem por que foram reprovados
2. **Auditoria**: Você pode provar que o modelo não é discriminatório
3. **Melhoria**: Clientes sabem o que fazer para melhorar (ex: "pague em dia por 3 meses")
4. **Debugar**: Se o modelo rejeitar um bom cliente, você vê quais features erraram
5. **Regulatório**: Atende exigências de transparência (LGPD, GDPR)
6. **Negócio**: Analistas podem questionar decisões e ajustar thresholds

## 📝 Exemplo Prático de Saída

### Entrada (Cliente):
```python
{
  "LIMIT_BAL": 50000,
  "AGE": 23,
  "PAY_0": 2,  # 2 meses de atraso
  "PAY_AMT1": 2000,
  "BILL_AMT1": 15000
}
```

### SHAP (Análise Técnica):
```
PAY_0 = 2 meses → SHAP: +0.32 (forte impacto negativo)
LIMIT_BAL = R$ 50k → SHAP: -0.08 (impacto positivo médio)
PAY_AMT1 = R$ 2k → SHAP: -0.05 (impacto positivo baixo)
```

### LLM (Explicação Humanizada):
```
🔴 Crédito Reprovado (85% de risco)

Principais motivos:
1. Você possui 2 meses de atraso nos pagamentos (PAY_0)
   - Este é o fator mais crítico na sua análise

2. Seu limite atual de R$ 50.000 é positivo, mas não 
   compensa o histórico de atrasos

3. Pagamento recente de R$ 2.000 demonstra esforço, 
   porém insuficiente para reverter o risco

💡 Recomendações:
- Regularize os pagamentos em atraso imediatamente
- Mantenha pagamentos pontuais por 3-6 meses
- Considere renegociar dívidas pendentes
```

## 🚀 Resumo

**SHAP** transforma o modelo de uma caixa-preta em um **sistema transparente e auditável**, fornecendo a base técnica que o **LLM (Ollama) transforma em linguagem humana** para os clientes.

```
Modelo ML    →  SHAP        →  LLM           →  Cliente
"85% risco"  →  "PAY_0=+0.32"  →  "2 meses de atraso"  →  "Entendi!"
```

É a combinação perfeita: **precisão técnica** (SHAP) + **comunicação humana** (LLM) 🎯

## 🔗 Referências

- [Paper Original SHAP](https://arxiv.org/abs/1705.07874)
- [Documentação SHAP](https://shap.readthedocs.io/)
- [Shapley Values (Teoria dos Jogos)](https://en.wikipedia.org/wiki/Shapley_value)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)

## 📚 Arquivos Relacionados no Projeto

- `src/explain.py` - Implementação do SHAP Explainer
- `src/prompts.py` - Construção de prompts com dados SHAP
- `src/llm.py` - Integração com Ollama para geração de texto
- `src/app.py` - Interface Streamlit com visualização SHAP
