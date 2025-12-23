"""
Sistema de Decisão de Crédito com IA
Aplicação Streamlit para análise de risco de crédito com explicações via LLM.
"""
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from typing import Tuple

from explain import compute_shap_single, extract_shap_factors
from prompts import build_credit_prompt
from llm import call_llm
import os


# Constantes
# Detecta se está rodando no Docker ou localmente
if os.path.exists("/app/models/modelo_credito.pkl"):
    MODEL_FILE = "/app/models/modelo_credito.pkl"  # Docker
else:
    MODEL_FILE = "../models/modelo_credito.pkl"  # Local
    
DEFAULT_THRESHOLD = 0.08
DEFAULT_TICKET_MEDIO = 100
DEFAULT_PREJUIZO = 1000


@st.cache_data(show_spinner=False, ttl=3600)
def generate_explanation(prompt: str) -> str:
    """Gera explicação usando LLM com cache de 1 hora."""
    response = call_llm(prompt)
    return response if response else "⚠️ Não foi possível gerar a explicação. Verifique se o Ollama está rodando."

# ----------------------
# CONFIGURAÇÃO INICIAL
# ----------------------
st.set_page_config(page_title="Credit Card Risk", layout="wide")

# CSS para remover menu hamburger e botão Deploy
st.markdown("""
    <style>
        /* Remover menu hamburger */
        #MainMenu {visibility: hidden;}
        
        /* Remover botão Deploy */
        .stDeployButton {display: none;}
        
        /* Remover footer "Made with Streamlit" */
        footer {visibility: hidden;}
        
        /* Remover "Manage app" button */
        header[data-testid="stHeader"] > div:nth-child(2) {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------
# CARREGAMENTO DO MODELO
# ----------------------
@st.cache_resource
def load_pipeline():
    """Carrega o pipeline de ML com cache."""
    try:
        return joblib.load(MODEL_FILE)
    except FileNotFoundError:
        st.error(f"❌ Arquivo {MODEL_FILE} não encontrado!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {str(e)}")
        st.stop()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     ticket_medio: float, prejuizo: float) -> Tuple:
    """
    Calcula métricas de negócio baseadas na matriz de confusão.

    Returns:
        Tuple com (total, taxa_aprovacao, taxa_inadimplencia, resultado_financeiro)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total = len(y_true)
    total_aprovados = tn + fn
    taxa_aprovacao = (total_aprovados / total) * 100 if total > 0 else 0
    taxa_inadimplencia = (fn / total_aprovados * 100) if total_aprovados > 0 else 0
    resultado_financeiro = (tn * ticket_medio) - (fn * prejuizo)

    return total, taxa_aprovacao, taxa_inadimplencia, resultado_financeiro, tn, fn, fp, tp

pipeline = load_pipeline()

# ----------------------
# PÁGINA INICIAL
# ----------------------
st.title("💳 Sistema de Decisão de Crédito")

st.markdown(""" Carregue seu arquivo CSV, ajuste o limiar de decisão,  
e receba **explicações claras** e **orientação de soluções** de crédito por um LLM local.""")

st.sidebar.header("1. Upload de Arquivos")
file = st.sidebar.file_uploader("Arraste seu CSV de Teste aqui", type="csv")


if file:
    df = pd.read_csv(file)

    st.subheader("Dados carregados")
    st.dataframe(df.head())

    # ----------------------
    # PARÂMETROS DE NEGÓCIO
    # ----------------------
    st.sidebar.markdown("---")
    st.sidebar.header("2. Parâmetros de Negócio")
    
    threshold = st.sidebar.slider(
        "Risco Máximo Aceitável (Corte)", 
        0.0, 1.0, DEFAULT_THRESHOLD, 0.01,
        help="Clientes com probabilidade acima deste valor serão reprovados"
    )
    ticket_medio = st.sidebar.number_input(
        "Lucro por Cliente (R$)", 
        value=DEFAULT_TICKET_MEDIO,
        help="Lucro médio esperado por cliente aprovado"
    )
    prejuizo_medio = st.sidebar.number_input(
        "Prejuízo por Calote (R$)", 
        value=DEFAULT_PREJUIZO,
        help="Prejuízo médio em caso de inadimplência"
    )

    # ----------------------
    # PREDIÇÃO E MÉTRICAS
    # ----------------------
    try:
        X_input = df.drop(columns=["default.payment.next.month", "ID"], errors="ignore")
        y_true = df['default.payment.next.month']

        # Predição
        probs = pipeline.predict_proba(X_input)[:, 1]
        df["default_probability"] = probs
        df["decision"] = np.where(probs >= threshold, "Reprovado", "Aprovado")

        st.subheader("Decisões do Modelo")
        st.dataframe(df, width='stretch')

        # Calcular métricas
        decisao_modelo = (probs >= threshold).astype(int)
        total, taxa_aprovacao, taxa_inadimplencia, resultado_financeiro, tn, fn, fp, tp = calculate_metrics(
            y_true, decisao_modelo, ticket_medio, prejuizo_medio
        )

    except Exception as e:
        st.error(f"❌ Erro ao processar dados: {str(e)}")
        st.stop()


    # ----------------------
    # KPIs
    # ----------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes Processados", f"{total:,}")
    c2.metric("Taxa de Aprovação", f"{taxa_aprovacao:.1f}%")
    c3.metric(
        "Inadimplência Real da Carteira", 
        f"{taxa_inadimplencia:.2f}%", 
        delta_color="inverse",
        help="Percentual de inadimplentes entre os aprovados"
    )
    c4.metric(
        "Resultado Financeiro", 
        f"R$ {resultado_financeiro:,.2f}",
        help=f"Lucro (R$ {tn * ticket_medio:,.0f}) - Prejuízo (R$ {fn * prejuizo_medio:,.0f})"
    )

    st.markdown("---")

    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.subheader("Distribuição de Risco dos Clientes")

        # Criar histograma usando Plotly GO (sem necessidade de px)
        fig_hist = go.Figure()

        # Histograma para clientes bons (0)
        fig_hist.add_trace(go.Histogram(
            x=probs[y_true == 0],
            name='Pagou',
            marker_color='green',
            opacity=0.6,
            nbinsx=50
        ))

        # Histograma para clientes ruins (1)
        fig_hist.add_trace(go.Histogram(
            x=probs[y_true == 1],
            name='Não Pagou',
            marker_color='red',
            opacity=0.6,
            nbinsx=50
        ))

        fig_hist.add_vline(x=threshold, line_dash="dash", annotation_text="CORTE")
        fig_hist.update_layout(
            title="Separação de Risco",
            xaxis_title="Probabilidade de Risco",
            yaxis_title="Quantidade",
            barmode='overlay'
        )
        st.plotly_chart(fig_hist, width='stretch')

    with col_r:
        st.subheader("Simulação de Carteira")
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Bons Aprovados', 'Maus Aprovados (Erro)', 'Rejeitados'], 
            values=[tn, fn, tp+fp], 
            hole=.4, 
            marker_colors=['green', 'red', 'gray']
        )])
        fig_pie.update_layout(showlegend=True)
        st.plotly_chart(fig_pie, width='stretch')

    # ----------------------
    # TABELA DETALHADA
    # ----------------------
    with st.expander("📊 Ver Dados Detalhados"):
        df_final = df.copy()
        df_final['Score_Risco'] = probs
        df_final['Decisao_Simulada'] = np.where(decisao_modelo == 1, 'REPROVADO', 'APROVADO')
        st.dataframe(df_final.head(100), width='stretch')


    # ----------------------
    # SELEÇÃO DE CLIENTE
    # ----------------------
    st.markdown("---")
    st.subheader("🔍 Análise Individual de Cliente")
    
    idx = st.selectbox(
        "Selecione o ID do Cliente",
        options=df.index,
        format_func=lambda x: f"Cliente {x}"
    )

    client = df.loc[idx]
    prob = client["default_probability"]
    decision = client["decision"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Decisão", decision, help="Aprovado ou Reprovado baseado no threshold")
    col2.metric("Probabilidade de Default", f"{prob:.1%}", help="Risco estimado pelo modelo")
    col3.metric("Threshold Atual", f"{threshold:.1%}", help="Limiar de decisão configurado")


    # ----------------------
    # EXPLICAÇÃO COM LLM
    # ----------------------
    if st.button("🤖 Gerar Explicação com IA", width='stretch'):
        with st.spinner("Calculando SHAP e gerando explicação..."):
            try:
                # Calcular SHAP values
                X_client = X_input.loc[[idx]]
                shap_values, X_transformed = compute_shap_single(pipeline, X_client)

                # Extrair principais fatores
                feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
                shap_df = extract_shap_factors(shap_values, feature_names, top_k=5)

                client_summary = shap_df.to_dict(orient="records")

                # Mostrar fatores
                st.subheader("📊 Principais Fatores (Baseados no Modelo)")
                st.dataframe(shap_df, width='stretch')

                # Gerar explicação com LLM
                prompt = build_credit_prompt(
                    decision=decision,
                    prob=prob,
                    threshold=threshold,
                    factors=client_summary
                )

                explanation = generate_explanation(prompt)

                st.subheader("📄 Explicação Personalizada")
                st.info(explanation)

            except Exception as e:
                st.error(f"❌ Erro ao gerar explicação: {str(e)}")
                st.write("Detalhes:", e)

else:
    st.info("👆 Por favor, faça upload de um arquivo CSV para começar.")
