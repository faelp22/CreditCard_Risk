"""
Script de Treinamento do Modelo de Risco de Crédito
Treina um modelo de classificação e salva o pipeline completo.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_prepare_data(file_path: str) -> tuple:
    """
    Carrega e prepara os dados para treinamento.

    Args:
        file_path: Caminho para o arquivo CSV

    Returns:
        X_train, X_test, y_train, y_test
    """
    print("📊 Carregando dados...")
    df = pd.read_csv(file_path)

    print(f"✅ Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")

    # Limpeza de dados (mesmo do notebook)
    print("🧹 Limpando dados...")
    df.loc[df.EDUCATION.isin([0, 5, 6]), 'EDUCATION'] = 4
    df.loc[df.MARRIAGE == 0, 'MARRIAGE'] = 3

    # Separar features e target
    X = df.drop(columns=["default.payment.next.month", "ID"], errors="ignore")
    y = df["default.payment.next.month"]

    print(f"✅ Features: {X.shape[1]}")
    print(f"✅ Target distribuição: {y.value_counts().to_dict()}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    print(f"✅ Train set: {X_train.shape[0]} amostras")
    print(f"✅ Test set: {X_test.shape[0]} amostras")

    return X_train, X_test, y_train, y_test


def create_pipeline() -> Pipeline:
    """
    Cria o pipeline de pré-processamento e modelo.

    Returns:
        Pipeline sklearn
    """
    print("🏗️ Criando pipeline...")

    # Preprocessador - apenas escalar (todas as features são numéricas)
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), slice(None))
        ],
        remainder='passthrough'
    )

    # Modelo
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Para lidar com desbalanceamento
    )

    # Pipeline completo
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    print("✅ Pipeline criado!")
    return pipeline


def train_model(pipeline: Pipeline, X_train, y_train):
    """
    Treina o modelo.
    """
    print("\n🚀 Iniciando treinamento...")
    pipeline.fit(X_train, y_train)
    print("✅ Treinamento concluído!")
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """
    Avalia o modelo no conjunto de teste.
    """
    print("\n📈 Avaliando modelo...")

    # Predições
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Métricas
    print("\n📊 RELATÓRIO DE CLASSIFICAÇÃO:")
    print(classification_report(y_test, y_pred, target_names=['Paga', 'Default']))

    print("\n📊 MATRIZ DE CONFUSÃO:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\n📊 ROC-AUC Score: {roc_auc:.4f}")

    # Visualizar matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Paga', 'Default'],
                yticklabels=['Paga', 'Default'])
    plt.title('Matriz de Confusão')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig('../reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Matriz de confusão salva em '../reports/confusion_matrix.png'")

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../reports/roc_curve.png', dpi=300, bbox_inches='tight')
    print("✅ Curva ROC salva em '../reports/roc_curve.png'")

    return roc_auc


def save_model(pipeline, filename: str = "../models/modelo_credito.pkl"):
    """
    Salva o pipeline treinado.
    """
    print(f"\n💾 Salvando modelo em '{filename}'...")
    joblib.dump(pipeline, filename)
    print("✅ Modelo salvo com sucesso!")
    print(f"📝 Relatório detalhado disponível em '../reports/MODEL_REPORT.md'")


def main():
    """
    Função principal de treinamento.
    """
    print("="*60)
    print("🎯 TREINAMENTO DO MODELO DE RISCO DE CRÉDITO")
    print("="*60)

    # 1. Carregar dados
    X_train, X_test, y_train, y_test = load_and_prepare_data("../data/UCI_Credit_Card.csv")

    # 2. Criar pipeline
    pipeline = create_pipeline()

    # 3. Treinar
    pipeline = train_model(pipeline, X_train, y_train)

    # 4. Avaliar
    roc_auc = evaluate_model(pipeline, X_test, y_test)

    # 5. Salvar
    save_model(pipeline)

    print("\n" + "="*60)
    print("✅ PROCESSO CONCLUÍDO COM SUCESSO!")
    print("="*60)
    print(f"📊 ROC-AUC Score: {roc_auc:.4f}")
    print(f"💾 Modelo salvo: models/modelo_credito.pkl")
    print(f"📈 Visualizações salvas: reports/confusion_matrix.png, reports/roc_curve.png")
    print(f"📝 Relatório completo: reports/MODEL_REPORT.md")
    print("\n🚀 Agora você pode executar o Streamlit com:")
    print("   cd src && streamlit run app.py")
    print("="*60)


if __name__ == "__main__":
    main()
