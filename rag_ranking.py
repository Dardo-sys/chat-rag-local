#!/usr/bin/env python3
# Autor: Dardo Esteban Nava <itdardonava@gmail.com>
# rag_ranking.py
# ============================================================================
# Ranking de recuperación + prompt de Q&A con análisis del RAG.
# Compartido por rag_query.py (uso) y run_rag_eval.py (evaluación).
#
# MOTIVACIÓN (decidida con el usuario): "dejar en segundo lugar las auditorías".
# En una carpeta heterogénea, los fragmentos de auditoría/veredicto que mencionan
# el tema puntúan alto en similaridad coseno y desplazan a la fuente primaria.
# Se aplica un ajuste auditable de ranking:
#   - PENALIZACIÓN a fragmentos de gestión/auditoría/control.
#   - PEQUEÑO BOOST a fragmentos de fuente primaria (papers/borradores/análisis).
# El veredicto de exactitud NO se toca: sólo se reordena qué entra en el top-K.
# ============================================================================

import os
import re

# Prioridad: fuente primaria de contenido (importancia +)
# Nota: adapta estos patrones al proyecto que indexes.
PRIMARY_PATTERNS = [
    "analisis_",               # análisis temáticos
    "MANUAL_", "TRATADO_", "GENESIS_",
]

# Auditoría / gestión / control (importancia -)
AUDIT_PATTERNS = [
    "Reportes_Analisis_Tematicos_DOCX",
    "_ORGANIZACION",
    "REINICIO_CONTROLADO",
    "auditor", "veredicto", "sugerenci", "correccion",
    "manifiesto", "logica", "plan_", "control", "indice",
]

BOOST_PRIMARY = 0.05
PENALTY_AUDIT = -0.12
TOKEN_BOOST = 0.05      # por palabra clave {no metadato} de la pregunta en el fragmento
METADATA_BOOST = 0.30   # intención de metadato + patrón de asignación del metadato

# palabras que delatan intención de metadato en la pregunta
METADATA_WORDS = ["autor", "author", "fecha", "date", "nombre", "name",
                  "titulo", "title", "email", "quien", "cuando"]

# Bonus a un definidor de metadato cuyo VALOR asignado sea "rico" (p. ej. un
# nombre completo de persona: "Autor: Ana Pérez"), frente a un valor
# corto o simbólico ("Autor: Ana & Co."). Evita que la simple
# presencia del patrón \author{}/autor: desplace a la fuente más informativa.
METADATA_VALUE_BOOST = 0.25


def _normalize_token(s):
    s = s.lower()
    accents = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n"}
    s = "".join(accents.get(c, c) for c in s)
    return re.sub(r"[^a-z0-9]", "", s)


def _has_metadata_intent(query):
    toks = [_normalize_token(t) for t in query.split()]
    return any(t in METADATA_WORDS for t in toks if t)


def _assignment_match(raw_text, meta_word):
    """Detecta si el fragmento DEFINE el metadato pedido (patrones de asignación):
       - LaTeX:  \author{Nombre}   o   author {
       - texto:  autor:  |  autor =  |  author :
    Tolerante a espacios entre la palabra y la llave."""
    aliases = {"autor": ["author"], "fecha": ["date"], "titulo": ["title"],
               "title": ["title"], "nombre": ["name"], "name": ["name"],
               "author": ["author"], "date": ["date"], "email": ["email"]}
    alts = [meta_word] + aliases.get(meta_word, [])
    for a in alts:
        simple = re.compile(r"[\\ ]?" + re.escape(a) + r"\s*[{:=]", re.IGNORECASE)
        if simple.search(raw_text):
            return True
    return False


def _name_value(raw_text):
    """Puntúa 0..1 la calidad del 'valor' tras un patrón de autor/name.
    - 1.0: nombre de PERSONA claro (>=2 palabras capitalizadas que no sean
      siglas ni estén separadas por '&'), p. ej. 'Ana Pérez'.
    - 0.5: valor nominal simple (1 palabra capitalizada) o nombre con '&'.
    - 0.0: sin valor claramente nominal / simbólico.
    Distingue 'Ana Pérez' (persona, 1.0) de 'Ana & Asociados'
    (equipo/marca, menor), para priorizar la fuente con el dato más completo."""
    m = re.search(r"(?:author|autor|name)\s*[{:=]\s*([^}\n]{0,90})", raw_text, re.I)
    if not m:
        return 0.0
    valor = m.group(1).strip().replace("\\", " ").strip()
    if not valor:
        return 0.0
    valor = valor.split("**")[0].strip()          # quitar markdown **Negrita**
    unidades = [u.strip() for u in re.split(r"[&,;]", valor) if u.strip()]
    if not unidades:
        return 0.0

    def capitalizadas(s):
        # palabras Capitalizadas que NO sean siglas (no todo-mayúsculas)
        return [w for w in re.findall(r"\b[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+\b", s)
                if not w.isupper()]

    total_caps = 0
    for u in unidades:
        caps = capitalizadas(u)
        total_caps += len(caps)
        if len(caps) >= 2:
            return 1.0
    if total_caps >= 1:
        return 0.5
    return 0.0


def lexical_boost(query, text, raw_text=None):
    """Refuerzo léxico para metadatos que el coseno semántico pierde
    (p. ej. el autor en un preámbulo LaTeX).
    - Metadato: si la pregunta pide un metadato (autor/fecha/título/…), premia de
      forma FUERTE al fragmento que DEFINE ese metadato (patrón \author{ / autor:).
    - Otras palabras clave de la pregunta: boost modesto por coincidencia exacta."""
    toks = [_normalize_token(t) for t in query.split()]
    stop = {"que", "es", "el", "la", "los", "las", "de", "del", "un", "una",
            "y", "o", "en", "a", "para", "cuando", "como", "se", "cual",
            "con", "por", "al", "su", "sus", "este", "esta", "son"}
    toks = [t for t in toks if t and t not in stop]

    t = _normalize_token(text)
    boost = 0.0

    meta_toks = [tok for tok in toks if tok in METADATA_WORDS]
    if meta_toks:
        rt = raw_text if raw_text is not None else text
        if any(_assignment_match(rt, mw) for mw in meta_toks):
            boost += METADATA_BOOST
        # si además el fragmento contiene el token-metadato, refuerza un poco
        for mw in meta_toks:
            if mw in t:
                boost += TOKEN_BOOST / 2

    # resto de palabras clave (no metadato): boost modesto por coincidencia exacta
    for tok in toks:
        if tok not in METADATA_WORDS and tok in t:
            boost += TOKEN_BOOST
    return boost


def _norm(ruta):
    return ruta.replace("\\", "/").lower()


def _match_any(ruta_lower, patterns):
    for p in patterns:
        if p.lower() in ruta_lower:
            return True
    return False


def adjust_score(score, ruta):
    """Devuelve el score ajustado por tipo de fuente (auditable)."""
    rl = _norm(ruta)
    adj = 0.0
    if _match_any(rl, AUDIT_PATTERNS):
        adj += PENALTY_AUDIT
    elif _match_any(rl, PRIMARY_PATTERNS):
        adj += BOOST_PRIMARY
    return score + adj, adj


def select_ranked(emb, qemb, src, chunks, K, query=""):
    """Recupera los K fragmentos mejor rankeados tras el ajuste.
    COMPOSICIÓN de dos pasadas (deduplicada por archivo):
      1) PASADA DE METADATO: si la pregunta pide un metadato (autor/fecha/…),
         fuerza los fragmentos que lo DEFINEN (patrón \author{ / autor:),
         ordenados por similaridad + boost de "valor rico" del metadato.
      2) PASADA SEMÁNTICA: similaridad coseno + ajuste por tipo de fuente
         + boost léxico para el resto del contenido.
    El resultado es la unión de ambas, garantizando que la fuente correcta
    (aunque tenga coseno bajo, p. ej. un preámbulo LaTeX con el nombre) no
    quede fuera de las citas."""
    candidates = []
    seen = {}
    used_idx = set()
    raw = emb @ qemb

    # --- COMPOSICIÓN: pasada de metadato (si aplica) ---
    is_meta = _has_metadata_intent(query)
    if is_meta:
        meta_toks = [_normalize_token(t) for t in query.split()
                     if _normalize_token(t) in METADATA_WORDS]
        # Para autor/nombre, el "autor de una publicación" vive en documentos de
        # texto (md/tex/txt), no en scripts .py/.js (licencias, "owned_by", etc.)
        # que disparan falsos positivos de assignment_match. Solo esos se suman.
        doc_exts = (".md", ".tex", ".txt", ".rst")
        by_file = {}
        for i, c in enumerate(chunks):
            if not src[i].lower().endswith(doc_exts):
                continue
            if any(_assignment_match(c, mw) for mw in meta_toks):
                rl = src[i]
                nv = _name_value(c)
                base_fila = (nv, float(raw[i]), i)
                if rl not in by_file or base_fila[:2] > by_file[rl][:2]:
                    by_file[rl] = base_fila
        # PRIORIZAR primero los definidores con valor rico (nombre completo),
        # y dentro de ese nivel por coseno: la fuente que mejor DEFINE el dato
        # no debe quedar fuera aunque puntúe bajo en similaridad.
        ranked = sorted(by_file.values(), key=lambda x: (-x[0], -x[1]))[:K]
        for nv, cos, i in ranked:
            candidates.append((i, 20.0 + cos + METADATA_VALUE_BOOST * nv,
                               METADATA_VALUE_BOOST * nv, cos))
            used_idx.add(i)

    # --- pasada semántica (complemento, deduplicado de metadato) ---
    order = np_argsort(-raw)
    for i in order:
        if i in used_idx:
            continue
        rl = _norm(src[i])
        is_primary = _match_any(rl, PRIMARY_PATTERNS)
        allow = 2 if is_primary else 1
        n = seen.get(src[i], 0)
        if n < allow:
            seen[src[i]] = n + 1
            base, adj = adjust_score(float(raw[i]), src[i])
            lex = lexical_boost(query, chunks[i], raw_text=chunks[i]) if query else 0.0
            candidates.append((i, base + lex, adj, lex))
        if len(candidates) >= K * 6:
            break
    candidates.sort(key=lambda x: -x[1])
    return candidates[:K]


def np_argsort(arr):
    import numpy as np
    return np.argsort(arr)


def hit_tuple(candidates, chunks, src, raw_sims, query=""):
    """convierte (idx, score_final, adj, lex) a listas de diccionarios."""
    out = []
    for idx, score_final, adj, lex in candidates:
        out.append({
            "texto": chunks[idx], "ruta": src[idx],
            "score_bruto": float(raw_sims[idx]), "score_ajustado": score_final,
            "ajuste": round(adj, 3), "boost_lexico": round(lex, 3), "query": query,
        })
    return out


# ---------------------------------------------------------------------------
# Prompt de Q&A con análisis
# ---------------------------------------------------------------------------
def build_prompt(question, hits):
    """hits: lista de dicts con 'texto','ruta'."""
    lines = [
        "Eres un asistente que responde EXCLUSIVAMENTE usando la información "
        "citada abajo (fragmentos del proyecto del usuario).",
        "",
        "Formato de respuesta en DOS partes:",
        "1) RESPUESTA: da el dato/hecho solicitado en forma directa.",
        "2) ANALISIS: explica qué significa, cómo encaja y por qué se llega a esa "
        "respuesta, razonando a partir de las citas. No agregues información que "
        "no esté respaldada por las citas.",
        "",
        "Para cada dato o afirmación usada, indica al final la RUTA del archivo de "
        "donde lo sacaste (formato: [ruta]).",
        "",
        "IMPORTANTE: ANTES de responder 'no-encontrado-en-el-indice', revisa TODAS las "
        "citas una por una. Si el dato aparece (aunque sea en inglés, con notación "
        "LaTeX, o en texto denso), respóndelo. Solo usa 'no-encontrado-en-el-indice' "
        "si ningún fragmento contiene el dato.\n",
        "Si la información solicitada NO aparece en las citas, responde exactamente "
        "'no-encontrado-en-el-indice' y no inventes nada.\n",
    ]
    lines.append("CITAS (fragmentos del proyecto):\n")
    for i, h in enumerate(hits, 1):
        lines.append(f"[Cita {i}] (fuente: {h['ruta']})\n{h['texto']}\n")
    lines.append(f"PREGUNTA: {question}\n")
    lines.append("RESPUESTA:")
    return "\n".join(lines)


def print_retrieved(hits, stream):
    print("\n=== FRAGMENTOS RECUPERADOS (auditable) ===")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {h['ruta']}  (coseno {h['score_bruto']:.3f} "
              f"/ ajuste {h['ajuste']:+.3f})")
        print(f"      {h['texto'][:120]}...")
