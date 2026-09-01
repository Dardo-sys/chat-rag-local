#!/usr/bin/env python3
# Autor: Dardo Nava (@Dardo-sys)
# rag_query.py
# ============================================================================
# Consulta Q&A sobre el índice RAG construido por rag_index.py.
# Recupera los fragmentos más relevantes con ranking ajustado (auditorías en
# segundo plano) y pide al LLM local responder con RESPUESTA + ANALISIS,
# citando las rutas de las fuentes.
#
# Uso:  EXP_MODEL=qwen2.5:3b python rag_query.py "tu pregunta"
#       (también en modo interactivo sin argumentos)
# ============================================================================

import os
import sys
import pickle
import argparse
import numpy as np

from sentence_transformers import SentenceTransformer
import runner
import config
import rag_ranking as RR
import rag_index as RIX

K = 5


def load_index(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Consulta Q&A sobre un indice RAG local.")
    parser.add_argument("--index", help="Nombre de indice (slug) o ruta .pkl", default=None)
    parser.add_argument("question", nargs="*", help="Pregunta (si se omite, modo interactivo)")
    args = parser.parse_args()

    slug = args.index or os.environ.get("RAG_INDEX")
    idx_path = RIX.index_path_for(slug) if slug else RIX.resolve_index_file()
    if not os.path.isfile(idx_path):
        print(f"[ERROR] No existe el indice: {idx_path}")
        print(f"Indices disponibles: {list(RIX.list_indices().keys())}")
        print("Crealo con:  python rag_index.py --folder <ruta>")
        sys.exit(1)

    print(f"Usando indice: {idx_path}")
    emb_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)
    index = load_index(idx_path)
    emb = np.asarray(index["emb"])
    chunks = index["chunks"]
    src = index["src"]

    if args.question:
        questions = [" ".join(args.question)]
    else:
        questions = []
        print("Modo interactivo. Escribe tu pregunta (ENTRE_VACIO=salir).")
        q = input("> ").strip()
        while q:
            questions.append(q)
            q = input("> ").strip()

    for question in questions:
        qemb = emb_model.encode([question], normalize_embeddings=True)[0]
        cands = RR.select_ranked(emb, qemb, src, chunks, K, query=question)
        raw = emb @ qemb
        hits = RR.hit_tuple(cands, chunks, src, raw, query=question)
        RR.print_retrieved(hits, print)

        prompt = RR.build_prompt(question, hits)
        res = runner.run_single(prompt)
        print("\n=== RESPUESTA DEL MODELO ===")
        print(res["response"].strip())
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
