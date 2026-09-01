#!/usr/bin/env python3
# Autor: Dardo Nava (@Dardo-sys)
# runner.py
# ============================================================================
# Ejecuta el pipeline de experimentacion contra un LLM real (via Ollama).
#
# PROTOCOLO DE HONESTIDAD:
#   - Generacion DETERMINISTA: temperature=0, seed fijo, mismo contexto.
#   - Se registran tiempos, conteo de tokens del modelo (prompt_eval_count /
#     eval_count) devuelto por Ollama, sin inventar valores.
#   - Cada llamada guarda el texto crudo de la respuesta.
# ============================================================================

import time
import json
import urllib.request

import config


def call_model(prompt: str, timeout: int = 300):
    """Llama al modelo via Ollama y devuelve respuesta + metadatos reales."""
    payload = {
        "model": config.MODEL,
        "prompt": prompt,
        "stream": False,
        "options": config.GENERATION,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        config.OLLAMA_URL, data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    return {
        "response": data.get("response", ""),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "eval_duration": data.get("eval_duration"),
        "prompt_eval_duration": data.get("prompt_eval_duration"),
        "llm_elapsed_s": round(elapsed, 3),
    }


def run_single(prompt: str):
    """Ejecuta una sola consulta y guarda TODOS los metadatos que el modelo
    devolvio, sin inventar ninguno."""
    try:
        return call_model(prompt)
    except Exception as e:
        return {
            "error": str(e),
            "response": None,
            "prompt_eval_count": None,
            "eval_count": None,
            "llm_elapsed_s": None,
        }
