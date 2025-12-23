"""
Módulo de construção de prompts para o LLM.
"""
from typing import List, Dict


def build_credit_prompt(
    decision: str,
    prob: float,
    threshold: float,
    factors: List[Dict],
    top_k: int = 2,
    language: str = "pt-BR"
) -> str:
    """
    Constrói um prompt de explicação em linguagem natural para decisões de crédito.

    Args:
        decision: Decisão tomada ("Aprovado" ou "Reprovado")
        prob: Probabilidade de default
        threshold: Threshold usado na decisão
        factors: Lista de dicionários com 'feature' e 'shap_value'
        top_k: Número de fatores principais a incluir
        language: Idioma da explicação (padrão: pt-BR)

    Returns:
        Prompt formatado para o LLM
    """
    # Ordenar fatores por impacto absoluto
    sorted_factors = sorted(
        factors,
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )

    main_factors = sorted_factors[:top_k]

    # Construir texto dos fatores
    factors_text = "\n".join([
        f"- {item['feature']} (impact: {item['shap_value']:.3f})"
        for item in main_factors
    ])

    # Template do prompt
    prompt = f"""
You are a credit risk analyst assistant.

Your task is to explain a credit decision using ONLY the information provided.
Do NOT mention models, SHAP, machine learning, scores, or probabilities explicitly.

Decision: {decision}

Main factors influencing the decision:
{factors_text}

Instructions:
1. First, clearly explain why the credit was {decision.lower()}, in simple language. 
   Explain in two topics the main reason.
2. Then, suggest practical and realistic actions the client could take to improve
   their chances of approval in the future.
3. Each suggestion must be directly related to the factors listed above. 
   Offer only 2 most important (based on the factors), short and direct suggestions.
4. Use a respectful, supportive, and customer-oriented tone.
5. Do NOT make promises of approval.

Write the explanation for a non-technical audience in Brazilian Portuguese, 
and elaborate a short answer, to not be exhausting for those who are reading. 
It needs to be short and clear.
"""

    return prompt.strip()
