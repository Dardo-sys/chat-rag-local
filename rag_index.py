#!/usr/bin/env python3
# Autor: Dardo Esteban Nava <itdardonava@gmail.com>
# rag_index.py
# ============================================================================
# Construye un índice RAG (embeddings) sobre los archivos textuales de un
# proyecto, excluyendo sesiones de IA y binarios/datasets pesados.
#
# PROTOCOLO DE HONESTIDAD:
#   - Excluye explícitamente .freebuff_sessions/.opencode_sessions/.glm_sessions
#   - Excluye binarios y datasets >1MB (ruido, no documentación).
#   - Presupuesto de indexación: RAG_MAX_MB (default 10 MB) para que la primera
#     corrida termine en tiempo razonable en una máquina hogareña; lo documenta.
#   - Guarda el índice a disco (output/indice_rag.pkl) para reutilizarlo.
# ============================================================================

import os
import re
import sys
import pickle
import time

from sentence_transformers import SentenceTransformer

DEFAULT_BASE = os.environ.get("RAG_FOLDER", "")  # sin ruta por defecto: se exige --folder / RAG_FOLDER
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
INDICES_DIR = os.path.join(OUT_DIR, "indices")


def resolve_base(argv=None):
    """Ruta a indexar: prioridad CLI --folder, luego env RAG_FOLDER."""
    if argv is None:
        argv = sys.argv[1:]
    if "--folder" in argv:
        i = argv.index("--folder")
        if i + 1 < len(argv):
            return argv[i + 1]
    base = os.environ.get("RAG_FOLDER") or DEFAULT_BASE
    if not base:
        raise SystemExit(
            "[ERROR] Debes indicar la carpeta a indexar:\n"
            "  python rag_index.py --folder <ruta>\n"
            "  o define RAG_FOLDER=<ruta>"
        )
    return base


def slugify(name):
    """Convierte el nombre de la carpeta raiz en un slug de archivo seguro."""
    s = re.sub(r"[^\w\-]+", "_", name)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s or "indice"


def resolve_index_file(argv=None):
    """Devuelve la ruta del .pkl donde se guarda el indice del proyecto actual."""
    base = resolve_base(argv)
    slug = slugify(os.path.basename(os.path.normpath(base)))
    return os.path.join(INDICES_DIR, f"{slug}.pkl")


def index_path_for(slug):
    """Devuelve la ruta .pkl para un slug dado (nombre de indice)."""
    if os.path.isfile(slug):
        return slug
    s = slugify(slug)
    return os.path.join(INDICES_DIR, f"{s}.pkl")


def list_indices():
    """Devuelve {slug: ruta_pkl} de los indices disponibles."""
    out = {}
    if not os.path.isdir(INDICES_DIR):
        return out
    for fn in sorted(os.listdir(INDICES_DIR)):
        if fn.endswith(".pkl"):
            out[fn[:-4]] = os.path.join(INDICES_DIR, fn)
    return out

TEXT_EXTS = {".md", ".py", ".txt", ".json", ".csv", ".tex", ".js"}
SKIP_DIRS = {".freebuff_sessions", ".opencode_sessions", ".glm_sessions", "__pycache__", ".git"}
MAX_FILE_MB = 1.0          # archivos mayores se consideran datasets/sesiones, se omiten
CHUNK_CHARS = 2000         # tamaño del fragmento (subido para que el dato no quede fuera)
OVERLAP = 200              # solapamiento entre fragmentos


def iter_doc_files(root):
    """Recorre el árbol y produce rutas de archivos textuales elegibles."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in TEXT_EXTS:
                continue
            full = os.path.join(dirpath, fn)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            if size == 0 or size > MAX_FILE_MB * 1024 * 1024:
                continue
            yield full, size


def chunk_text(text):
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= CHUNK_CHARS:
        return [text]
    words = text.split(" ")
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        buf = []
        length = 0
        j = i
        while j < n:
            w = words[j]
            add = 1 + len(w) if buf else len(w)
            if length + add > CHUNK_CHARS:
                break
            buf.append(w)
            length += add
            j += 1
        chunks.append(" ".join(buf))
        # avanzar cubriendo CHUNK_CHARS - OVERLAP, siempre mínimo 1 palabra
        advance_chars = 0
        k = i
        target = CHUNK_CHARS - OVERLAP
        while k < j and advance_chars + (1 + len(words[k]) if k > i else len(words[k])) < target:
            advance_chars += (1 + len(words[k])) if k > i else len(words[k])
            k += 1
        next_i = k if k > i else i + 1
        if next_i <= i:
            next_i = i + 1
        i = next_i
    return chunks


def main():
    os.makedirs(INDICES_DIR, exist_ok=True)
    BASE = resolve_base()
    INDEX_FILE = resolve_index_file()
    print(f"Indexando carpeta:\n  {BASE}\n")
    print("Cargando modelo de embeddings (multilingüe, local)...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)
    print("Modelo listo.\n")

    # recolectar archivos elegibles
    files = list(iter_doc_files(BASE))
    print(f"Archivos textuales elegibles (sin sesiones/bits/datasets): {len(files)}")

    # aplicar presupuesto de MB (ordenando por ruta para reproducibilidad)
    max_mb = float(os.environ.get("RAG_MAX_MB", "10"))
    files.sort(key=lambda x: x[0])
    budget_bytes = max_mb * 1024 * 1024
    chosen = []
    used = 0
    for full, size in files:
        if used + size > budget_bytes:
            break
        chosen.append(full)
        used += size
    print(f"Presupuesto: {max_mb} MB -> indexando {len(chosen)} archivos (~{used/1048576:.1f} MB).\n")

    # trocear y coleccionar metadatos
    chunks = []
    src = []        # (id -> ruta)
    for full in chosen:
        try:
            with open(full, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            print(f"  (omitido por error: {e}) {full}")
            continue
        rel = os.path.relpath(full, BASE)
        for c in chunk_text(text):
            chunks.append(c)
            src.append(rel)

    if not chunks:
        print("No se generaron fragmentos.")
        return
    print(f"Fragmentos totales: {len(chunks)}")

    # codificar en lote (rápido con batch) y guardar
    t0 = time.time()
    emb = model.encode(chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    dt = time.time() - t0
    print(f"Embeddings generados en {dt:.1f}s ({len(emb)} vectores)")

    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"emb": emb, "chunks": chunks, "src": src, "archivo_base": BASE,
                     "modelo_emb": "paraphrase-multilingual-MiniLM-L12-v2"}, f)
    print(f"\nÍndice guardado en: {INDEX_FILE}")
    print(f"  {len(src)} fragmentos · {len(set(src))} archivos referenciados")


if __name__ == "__main__":
    main()
