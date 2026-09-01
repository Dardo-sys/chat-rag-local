#!/usr/bin/env python3
# Autor: Dardo Esteban Nava <itdardonava@gmail.com>
# config.py
# ============================================================================
# Configuracion central del experimento de compresion semantica de prompts.
#
# PROTOCOLO DE HONESTIDAD:
#   - Todos los parametros que afectan el resultado viven AQUI y se registran
#     en el reporte, de modo que el experimento sea reproducible y auditable.
#   - No se adulteran muestras, calculos ni resultados. Lo que no puede
#     determinarse se marca como `None` en el reporte, nunca se inventa.
# ============================================================================

import os
import json
import hashlib

# --- Modelo de IA usado para el experimento -------------------------------
# Ejecutable en cualquier modelo servido por Ollama.
MODEL = os.environ.get("EXP_MODEL", "smollm2:1.7b")


def set_model(name: str):
    """Permite cambiar el modelo en caliente (p. ej. desde la interfaz web).
    Acepta 'nombre' o 'nombre:tamaño' como los reporta Ollama."""
    global MODEL
    MODEL = name

# Endpoint de Ollama (servidor local levantado con los modelos de disco F:).
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_URL = f"http://{OLLAMA_HOST}/api/generate"

# Parametros de generacion FIJOS (determinismo para honestidad):
# temperature=0 => salida determinista (con el mismo seed y contexto).
GENERATION = {
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 42,
    "num_predict": 600,
}

# Directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RAW_DIST_DIR = None  # reservado: datos externos de referencia (sin usar aqui)

# El modelo puede responder distinto segun la configuracion; registramos
# siempre el hash de la configuracion para auditar que nada cambio a mitad
# de corrida.
def config_fingerprint() -> str:
    blob = {
        "model": MODEL,
        "llm_url": OLLAMA_URL,
        "generation": GENERATION,
        "protocol_version": "nivel-A-v1",
    }
    raw = json.dumps(blob, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]
