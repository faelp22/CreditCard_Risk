"""
Módulo de explicabilidade usando SHAP (SHapley Additive exPlanations).
"""
import shap
import pandas as pd
from typing import Tuple


def compute_shap_single(pipeline, X_client: pd.DataFrame) -> Tuple[any, pd.DataFrame]:
    """
    Calcula os valores SHAP para um único cliente.

    Args:
        pipeline: Pipeline sklearn com preprocessor e classifier
        X_client: DataFrame com os dados do cliente (uma linha)

    Returns:
        Tuple contendo:
            - shap_values: Valores SHAP calculados
            - X_transformed_df: DataFrame transformado com feature names
    """
    model = pipeline.named_steps["classifier"]
    preprocess = pipeline.named_steps["preprocessor"]

    # Transformar dados do cliente
    X_transformed = preprocess.transform(X_client)
    feature_names = preprocess.get_feature_names_out()

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names
    )

    # Calcular SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(
        X_transformed_df,
        check_additivity=False
    )

    return shap_values, X_transformed_df


def extract_shap_factors(
    shap_values,
    feature_names,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Extrai os principais fatores baseados nos valores SHAP.

    Args:
        shap_values: Valores SHAP calculados
        feature_names: Nomes das features
        top_k: Número de features mais importantes a retornar

    Returns:
        DataFrame com features, valores SHAP e importância absoluta
    """
    # Tratar diferentes formatos de SHAP values
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]  # Classe positiva
    elif len(shap_values.shape) == 3:
        shap_vals = shap_values[0, :, 1]  # Classe positiva
    else:
        shap_vals = shap_values[0]

    # Criar DataFrame ordenado
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_vals
    })

    return (
        shap_df
        .assign(abs_shap=lambda x: x.shap_value.abs())
        .sort_values("abs_shap", ascending=False)
        .head(top_k)
        .drop(columns="abs_shap")
    )
