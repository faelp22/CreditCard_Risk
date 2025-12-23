"""
Módulo de integração com LLM (Ollama) para geração de explicações de crédito.
"""
import requests
from typing import Optional
import logging
import json

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações do Ollama
OLLAMA_URL = "http://192.168.33.112:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"  # Modelo leve (379 MB, 494M parâmetros)
DEFAULT_TIMEOUT = 120  # segundos
MAX_RETRIES = 2


def call_llm(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 500,
    timeout: int = DEFAULT_TIMEOUT
) -> Optional[str]:
    """
    Chama o LLM local (Ollama) para gerar explicações.

    Args:
        prompt: O prompt para o LLM
        temperature: Controla a criatividade (0.0 = determinístico, 1.0 = criativo)
        max_tokens: Número máximo de tokens na resposta
        timeout: Tempo máximo de espera em segundos

    Returns:
        Resposta do LLM ou None em caso de erro
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature
        }
    }

    print("LLM Payload:")
    print("-----------------------------------------------------")
    print(json.dumps(payload))
    print("-----------------------------------------------------")

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Chamando LLM (tentativa {attempt + 1}/{MAX_RETRIES})...")

            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=timeout
            )

            response.raise_for_status()
            result = response.json().get("response", "")

            print("LLM Response:")
            print("-----------------------------------------------------")
            print(response.text)
            print("-----------------------------------------------------")

            if result:
                logger.info("Resposta do LLM recebida com sucesso.")
                return result
            else:
                logger.warning("LLM retornou resposta vazia.")

        except requests.exceptions.Timeout:
            logger.error(f"Timeout ao chamar LLM (tentativa {attempt + 1}/{MAX_RETRIES})")

        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao chamar LLM: {str(e)}")

        except Exception as e:
            logger.error(f"Erro inesperado: {str(e)}")

    logger.error("Falha ao obter resposta do LLM após todas as tentativas.")
    return None
