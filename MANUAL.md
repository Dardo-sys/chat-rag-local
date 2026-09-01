# Manual del Chat RAG

## Preguntas de ejemplo

- Pega la ruta de una carpeta con tus archivos, pulsa **Indexar**.

## Flujo diario

1. Doble clic en `chat_web.bat`.
2. Se abre una consola: presiona Enter (usa el indice por defecto) o escribe
   el nombre del proyecto.
3. Se abre el navegador en `http://127.0.0.1:8000/`.
4. En el selector **Proyecto activo** eliges con que carpeta hablar.
5. En **Modelo IA** eliges el modelo LLM (p. ej. `gemma2:2b` rapido,
   `gemma3:4b` mas completo).
6. En **Fuentes/consulta** eliges cuantos fragmentos usar (1-15).
7. Escribes una pregunta y Enter.

## Como leo una respuesta

Cada respuesta muestra:

- Los **fragmentos recuperados** con su ruta, coseno y ajuste.
- La **respuesta** del modelo.
- Al pie: cuantas **fuentes**, el **tiempo** que tardo y el **modelo** usado.
- El nombre de cada fuente es un **link**: te abre el archivo real en su carpeta.

## Errores comunes

| Sintoma | Causa / solucion |
|---|---|
| "No se encontro Python" | Instala Python y marca la opcion "Add to PATH". |
| "[AVISO] No se detecto Ollama" | Abre Ollama (trayectoria) antes de preguntar. |
| No existe el indice | Primero indexa: `python rag_index.py --folder <ruta>` |
| La respuesta se ve cortada | El limite de tokens esta en `config.py` (`num_predict`); subelo. |
| Modelo muy lento | Usa un modelo 2B (menos parametros) en el selector. |

## Mas carpeta a indexar

Puedes indexar todas las carpetas que quieras; cada una es un proyecto
independiente y no se mezclan los historiales.

```
python rag_index.py --folder "C:\otra\carpeta"
```

---
Autor: Dardo Nava ([@Dardo-sys](https://github.com/Dardo-sys))
