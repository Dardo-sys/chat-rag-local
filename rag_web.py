#!/usr/bin/env python3
# Autor: Dardo Nava (@Dardo-sys)
# rag_web.py
# ============================================================================
# Interfaz web "todo en uno" para el RAG local.
#   - Pegar la ruta de cualquier carpeta y INDEXARLA desde el navegador.
#   - Elegir entre los indices creados (varios proyectos) sin reiniciar.
#   - Hablar con el proyecto activo en un chat, citando fuentes.
# Solo usa la biblioteca estandar de Python + sentence_transformers.
#
# API HTTP:
#   GET  /                       -> pagina del chat
#   GET  /api/indices            -> indices disponibles (slug -> ruta)
#   GET  /api/status             -> estado de indexacion ()estado/err/log)
#   GET  /api/active             -> indice activo + carpetas
#   POST /api/index   {folder}   -> indexa una carpeta (en segundo plano)
#   POST /api/select  {index}    -> activa un indice
#   POST /api/ask     {question} -> pregunta al RAG con el indice activo
# ============================================================================

import json
import os
import sys
import time
import pickle
import threading
import urllib.parse
import urllib.request
import traceback

import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import runner
import config
import rag_ranking as RR
import rag_index as RIX

BASE = os.path.dirname(os.path.abspath(__file__))
HOST = "127.0.0.1"
PORT = int(os.environ.get("RAG_PORT", "8000"))
K = int(os.environ.get("RAG_K", "5"))  # fuentes por consulta (editable desde la UI)
_K_LOCK = threading.Lock()
EMB_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def set_k(n):
    """Cambia en caliente cuantas fuentes se recuperan por consulta."""
    global K
    n = max(1, min(15, int(n)))
    with _K_LOCK:
        K = n
    return K

# ---- recursos cargados una sola vez (bajo demanda) ------------------------
_ENC = None            # modelo de embeddings (lazy)
_ENC_LOCK = threading.Lock()
_ACTIVE = None         # dict índice activo
_ACTIVE_LOCK = threading.Lock()
_INDEXING = {"active": False, "slug": None, "done": False,
             "error": None, "log": []}   # estado de indexación


def _get_encoder():
    """Carga el modelo de embeddings una sola vez (thread-safe)."""
    global _ENC
    if _ENC is None:
        with _ENC_LOCK:
            if _ENC is None:
                from sentence_transformers import SentenceTransformer
                _ENC = SentenceTransformer(EMB_MODEL_NAME, local_files_only=True)
    return _ENC


def _load_index(index_file):
    """Carga un indice desde el .pkl (no toca embeddings)."""
    with open(index_file, "rb") as f:
        idx = pickle.load(f)
    return {
        "emb": np.asarray(idx["emb"]),
        "chunks": idx["chunks"],
        "src": idx["src"],
        "archivo_base": idx.get("archivo_base", "?"),
        "path": index_file,
    }


def _set_active(index_file, slug):
    global _ACTIVE
    with _ACTIVE_LOCK:
        _ACTIVE = _load_index(index_file)
        _ACTIVE["slug"] = slug


def get_active():
    with _ACTIVE_LOCK:
        return _ACTIVE


def _default_index():
    """Si no hay indice activo, usa el del proyecto por defecto si existe."""
    d = RIX.resolve_index_file()
    if os.path.isfile(d):
        slug = d.split(os.sep)[-1][:-4]
        _set_active(d, slug)
    else:
        avail = RIX.list_indices()
        if avail:
            first = min(avail)
            _set_active(avail[first], first)


def do_index(folder):
    """Indexa una carpeta en un hilo; registra un estado simple en _INDEXING.
    El texto de progreso se imprime en la consola del servidor (para no
    interferir con isatty/progress-bars de bibliotecas)."""
    def work():
        _INDEXING.update(active=True, done=False, error=None, log=[])
        try:
            _run_index_core(folder)
            _INDEXING["slug"] = RIX.slugify(os.path.basename(os.path.normpath(folder)))
            _INDEXING["done"] = True
        except Exception as e:
            _INDEXING["error"] = str(e)
            _INDEXING["log"].append("ERROR: " + traceback.format_exc())
        finally:
            _INDEXING["active"] = False

    threading.Thread(target=work, daemon=True).start()


def _run_index_core(folder):
    """Replica la logica de rag_index.main para una carpeta dada, devolviendo archivo."""
    import re
    import time
    index_file = RIX.index_path_for(RIX.slugify(os.path.basename(os.path.normpath(folder))))
    os.makedirs(RIX.INDICES_DIR, exist_ok=True)
    print(f"Indexando carpeta:\n  {folder}\n")
    enc = _get_encoder()
    files = list(RIX.iter_doc_files(folder))
    max_mb = float(os.environ.get("RAG_MAX_MB", "10"))
    files.sort(key=lambda x: x[0])
    budget = max_mb * 1024 * 1024
    chosen, used = [], 0
    for full, size in files:
        if used + size > budget:
            break
        chosen.append(full)
        used += size
    print(f"Presupuesto {max_mb} MB -> {len(chosen)} archivos ({used/1048576:.1f} MB)")
    chunks, src = [], []
    for full in chosen:
        try:
            with open(full, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            print(f"(omitido por error: {e}) {full}")
            continue
        rel = os.path.relpath(full, folder)
        for c in RIX.chunk_text(text):
            chunks.append(c)
            src.append(rel)
    if not chunks:
        raise RuntimeError("La carpeta no genero fragmentos textuales (sin .md/.txt/.tex/etc).")
    print(f"Fragmentos: {len(chunks)}")
    t0 = time.time()
    emb = enc.encode(chunks, batch_size=64, normalize_embeddings=True)
    print(f"Embeddings en {time.time()-t0:.1f}s")
    with open(index_file, "wb") as f:
        pickle.dump({"emb": emb, "chunks": chunks, "src": src,
                     "archivo_base": folder, "modelo_emb": EMB_MODEL_NAME}, f)
    print(f"Indice guardado: {index_file}")


# ============================================================================
# PAGINA WEB
# ============================================================================
HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chat RAG — tus archivos</title>
<style>
  :root{
    --bg:#0f1220; --panel:#171b2e; --panel2:#1d2340; --panel3:#222a4d;
    --line:#2a3155; --txt:#e8e9f3; --sub:#9aa0c0;
    --accent:#6c7bff; --accent2:#4ade80; --warn:#f5b954; --err:#ff7a7a;
    --host:#282f52; --user:#3b4dd8;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       background:radial-gradient(1200px 600px at 20% -10%, #232a55 0%, var(--bg) 55%);
       color:var(--txt);height:100vh;display:flex;flex-direction:column}
  header{padding:12px 20px;border-bottom:1px solid var(--line);
         display:flex;align-items:center;gap:12px;flex-wrap:wrap;
         background:rgba(23,27,46,.6);backdrop-filter:blur(6px)}
  header .logo{width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,var(--accent),var(--accent2));
       display:flex;align-items:center;justify-content:center;font-weight:800;color:#0b0f1e}
  header h1{font-size:15px;margin:0}
  header .sub{font-size:11px;color:var(--sub);line-height:1.35}
  .badge{margin-left:auto;font-size:11px;color:var(--accent2);border:1px solid var(--line);
       padding:4px 10px;border-radius:99px;background:var(--panel);white-space:nowrap}
  .configbar{padding:12px max(16px,calc((100% - 780px)/2));border-bottom:1px solid var(--line);
       background:rgba(23,27,46,.4);display:flex;flex-direction:column;gap:10px}
  .configbar .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  .configbar .lbl{font-size:11px;color:var(--sub);font-weight:700;min-width:110px}
  input[type=text],select{background:var(--panel);color:var(--txt);border:1px solid var(--line);
       border-radius:9px;padding:8px 10px;font:inherit;font-size:13px;outline:none;flex:1;min-width:160px}
  input[type=text]:focus,select:focus{border-color:var(--accent)}
  select{flex:0 1 auto;max-width:340px}
  .btn{border:none;border-radius:9px;padding:8px 14px;font:inherit;font-weight:700;font-size:13px;
       cursor:pointer;background:linear-gradient(135deg,var(--accent),#5048d0);color:#fff;white-space:nowrap}
  .btn.ghost{background:var(--panel3);border:1px solid var(--line)}
  .btn:disabled{opacity:.5;cursor:not-allowed}
  .statline{font-size:11px;color:var(--sub)}
  .statline.ok{color:var(--accent2)}
  .statline.err{color:var(--err)}
  #chat{flex:1;overflow-y:auto;padding:20px max(16px,calc((100% - 780px)/2));display:flex;
        flex-direction:column;gap:16px}
  .msg{max-width:min(620px,88%);display:flex;flex-direction:column;gap:8px}
  .msg .who{font-size:11px;color:var(--sub);font-weight:600;letter-spacing:.4px}
  .msg .bubble{padding:12px 15px;border-radius:16px;line-height:1.5;white-space:pre-wrap;
        word-wrap:break-word;font-size:14px}
  .msg.user{align-self:flex-end;align-items:flex-end}
  .msg.user .bubble{background:var(--user);border-bottom-right-radius:4px}
  .msg.bot{align-self:flex-start;align-items:flex-start}
  .msg.bot .bubble{background:var(--panel2);border:1px solid var(--line);border-bottom-left-radius:4px;width:100%}
  .sources{display:flex;flex-direction:column;gap:8px;margin-top:4px}
  .src{border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;
       background:var(--panel);padding:8px 11px}
  .src .ruta{font-size:11px;color:var(--accent);font-family:ui-monospace,Consolas,monospace;word-break:break-all;text-decoration:none}
  .src a.ruta:hover{text-decoration:underline;opacity:.8}
  .src .meta{font-size:10px;color:var(--sub);margin:3px 0}
  .src .txt{font-size:12px;color:var(--txt);opacity:.85}
  .meta-line{font-size:11px;color:var(--sub)}
  .typing{display:inline-flex;gap:5px;padding:12px 15px;border-radius:16px;background:var(--panel2);
       border:1px solid var(--line);align-items:center}
  .typing span{width:8px;height:8px;border-radius:50%;background:var(--accent);animation:blink 1.2s infinite}
  .typing span:nth-child(2){animation-delay:.2s}.typing span:nth-child(3){animation-delay:.4s}
  @keyframes blink{0%,80%,100%{opacity:.2}40%{opacity:1}}
  #inputbar{display:flex;gap:10px;padding:14px max(16px,calc((100% - 780px)/2));
       border-top:1px solid var(--line);background:rgba(23,27,46,.7);backdrop-filter:blur(6px);align-items:flex-end}
  #inp{flex:1;resize:none;background:var(--panel);color:var(--txt);border:1px solid var(--line);
       border-radius:14px;padding:12px 15px;font:inherit;outline:none;min-height:44px;max-height:140px}
  #inp:focus{border-color:var(--accent)}
  #send{border:none;border-radius:14px;padding:12px 22px;font:inherit;font-weight:700;cursor:pointer;
       background:linear-gradient(135deg,var(--accent),#5048d0);color:#fff}
  #send:disabled{opacity:.5;cursor:not-allowed}
</style>
</head>
<body>
<header>
  <div class="logo">R</div>
  <div>
    <h1>Chat RAG — habla con tus archivos</h1>
    <div class="sub" id="proyecto">Sin indice activo</div>
  </div>
  <span class="badge">● 100% local</span>
</header>

<div class="configbar">
  <div class="row">
    <span class="lbl">Proyecto activo</span>
    <select id="selIndex"></select>
    <button class="btn ghost" id="btnRefresh" title="Refrescar lista">⟳</button>
  </div>
  <div class="row">
    <span class="lbl">Indexar carpeta</span>
    <input type="text" id="folder" placeholder="Pega la ruta de la carpeta, ej: C:\\Users\\tuUsuario\\Documentos\\mi-proyecto">
    <button class="btn" id="btnIndex">Indexar</button>
  </div>
  <div class="row">
    <span class="lbl">Modelo IA</span>
    <select id="selModel"></select>
    <button class="btn ghost" id="btnModel" title="Aplicar modelo">Aplicar</button>
  </div>
  <div class="row">
    <span class="lbl">Fuentes/consulta</span>
    <input type="number" id="kInput" min="1" max="15" value="5" style="flex:0 1 90px">
    <button class="btn ghost" id="btnK" title="Aplicar cantidad de fuentes">Aplicar</button>
  </div>
  <div class="statline" id="stats"></div>
</div>

<main id="chat"></main>

<footer id="inputbar">
  <textarea id="inp" rows="1" placeholder="Escribe tu pregunta... (Enter enviar, Shift+Enter salto de linea)"></textarea>
  <button id="send">Enviar</button>
</footer>

<script>
const chat=$('chat'), inp=$('inp'), send=$('send');
const selIndex=$('selIndex'), btnRefresh=$('btnRefresh'), folder=$('folder'), btnIndex=$('btnIndex');
const selModel=$('selModel'), btnModel=$('btnModel');
const kInput=$('kInput'), btnK=$('btnK');
const stats=$('stats'), proyecto=$('proyecto');
function $(id){return document.getElementById(id);}
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function jget(u){const r=await fetch(u);return r.json();}
async function jpost(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});return r.json();}

function addMsg(who,text){
  const n=document.createElement('div'); n.className='msg '+who;
  n.innerHTML=`<div class="who">${esc(who==='user'?'Tu':'Asistente')}</div><div class="bubble">${esc(text)}</div>`;
  chat.appendChild(n); chat.scrollTop=chat.scrollHeight; return n;
}
function addTyping(){
  const n=document.createElement('div'); n.className='msg bot';
  n.innerHTML=`<div class="who">Asistente</div><div class="typing"><span></span><span></span><span></span></div>`;
  chat.appendChild(n); chat.scrollTop=chat.scrollHeight; return n;
}
function addWelcome(){
  const n=document.createElement('div'); n.className='msg bot';
  n.innerHTML=`<div class="who">Asistente</div><div class="bubble">Hola! Soy el asistente que responde sobre tus archivos.
  Pega arriba la ruta de la carpeta con tus documentos y pulsa "Indexar".
  Luego elige el proyecto en el selector y preguntame lo que quieras.
  Respondo solo con tu documentacion y cito las fuentes de donde saco la respuesta.</div>`;
  chat.appendChild(n); chat.scrollTop=chat.scrollHeight;
}

const historial=new Map();
let slugActual='__inicio__';
function guardarChat(){ if(slugActual) historial.set(slugActual, chat.innerHTML); }
function cargarChat(slug){
  chat.innerHTML = historial.get(slug) || '';
  if(!chat.children.length) addWelcome();
  chat.scrollTop=chat.scrollHeight;
}
async function cambiarIndice(slug){
  if(!slug) return;
  guardarChat();
  await jpost('/api/select',{index:slug});
  slugActual=slug;
  await refreshActive();
  cargarChat(slug);
}

async function refreshIndices(selectSlug){
  const data=await jget('/api/indices');
  const keys=Object.keys(data);
  selIndex.innerHTML='';
  if(!keys.length){
    const o=document.createElement('option'); o.value=''; o.textContent='(sin indices - indexa una carpeta)'; selIndex.appendChild(o);
  }else{
    keys.forEach(s=>{const o=document.createElement('option'); o.value=s; o.textContent=s; selIndex.appendChild(o);});
    if(selectSlug&&keys.includes(selectSlug)) selIndex.value=selectSlug;
  }
}
async function refreshActive(){
  const a=await jget('/api/active');
  proyecto.textContent = a && a.archivo_base ? ('Proyecto: '+a.archivo_base) : 'Sin indice activo';
}
async function refreshModels(){
  const m=await jget('/api/models');
  selModel.innerHTML='';
  const grupos={};
  (m.modelos||[]).forEach(x=>{
    const g=grupos[x.family]||(grupos[x.family]=[]);
    g.push(x);
  });
  Object.keys(grupos).forEach(fam=>{
    const og=document.createElement('optgroup'); og.label=fam.toUpperCase();
    grupos[fam].forEach(x=>{
      const o=document.createElement('option'); o.value=x.name;
      o.textContent=`${x.name}  (${x.parametros}B, ${x.gb}${x.quant?' · '+x.quant:''})`;
      og.appendChild(o);
    });
    selModel.appendChild(og);
  });
  const modelos=(m.modelos||[]).map(x=>x.name);
  if(m.activo && modelos.includes(m.activo)) selModel.value=m.activo;
  else if(m.activo){ const o=document.createElement('option'); o.value=m.activo; o.textContent=m.activo+' (activo)'; selModel.prepend(o); selModel.value=m.activo; }
}
async function applyModel(){
  const mv=selModel.value; if(!mv) return;
  const r=await jpost('/api/model',{model:mv});
  if(r.ok){ stats.textContent='Modelo activo: '+r.activo; stats.className='statline ok'; }
  else { stats.textContent='Error: '+(r.error||'desconocido'); stats.className='statline err'; }
}
async function applyK(){
  let v=parseInt(kInput.value,10); if(isNaN(v)||v<1||v>15){ stats.textContent='K debe estar entre 1 y 15'; stats.className='statline err'; return; }
  const r=await jpost('/api/k',{k:v});
  if(r.ok){ kInput.value=r.k; stats.textContent='Fuentes por consulta: '+r.k; stats.className='statline ok'; }
  else { stats.textContent='Error: '+(r.error||'desconocido'); stats.className='statline err'; }
}
async function refreshConfig(){
  const c=await jget('/api/config');
  if(c.k!=null){ kInput.value=c.k; }
}

function bubbleOf(n){return n.querySelector('.bubble');}

async function ask(){
  const q=inp.value.trim(); if(!q) return;
  addMsg('user',q); inp.value=''; autosize();
  const t=addTyping(); busy(true);
  try{
    const data=await jpost('/api/ask',{question:q});
    t.remove();
    if(data.error){ addMsg('bot','[error] '+data.error); return; }
    addMsg('bot', data.answer || '(sin respuesta)');
    const b=addMsg('bot',''); const bb=bubbleOf(b); bb.innerHTML='';
    const sc=document.createElement('div'); sc.className='sources';
    (data.sources||[]).forEach(s=>{
      const d=document.createElement('div'); d.className='src';
      const link=document.createElement('a');
      const p=s.abs.split('\\\\').join('/');
      link.href='file:///'+encodeURI(p);
      link.target='_blank'; link.rel='noopener'; link.className='ruta';
      link.textContent=s.ruta; link.title='Abrir en su carpeta';
      const metaDiv=document.createElement('div'); metaDiv.className='meta';
      metaDiv.textContent=`coseno ${s.coseno} · ajuste ${s.ajuste}`;
      const txt=document.createElement('div'); txt.className='txt';
      txt.textContent=(s.texto||'').slice(0,220);
      d.appendChild(link); d.appendChild(metaDiv); d.appendChild(txt);
      sc.appendChild(d);
    });
    bb.appendChild(sc);
    const meta=document.createElement('div'); meta.className='meta-line';
    const nf=(data.sources||[]).length;
    meta.textContent=`${nf} fuente(s) recuperadas` + (data.tiempo_s!=null?` · tardó ${data.tiempo_s}s`:'') + (data.modelo?` · modelo ${data.modelo}`:'');
    bb.appendChild(meta);
  }catch(e){ t.remove(); addMsg('bot','[error de conexion] '+e); }
  finally{ busy(false); inp.focus(); }
}

async function doIndex(){
  const f=folder.value.trim(); if(!f) return;
  stats.textContent='Indexando en segundo plano...'; stats.className='statline';
  btnIndex.disabled=true;
  const r=await jpost('/api/index',{folder:f});
  if(r.error){ stats.textContent='Error: '+r.error; stats.className='statline err'; btnIndex.disabled=false; return; }
  pollIndex(r.slug);
}
async function pollIndex(slug){
  const a=await jget('/api/status');
  if(a.active){ stats.textContent='Indexando '+(a.slug||'')+'... '+(a.log&&a.log.length?a.log[a.log.length-1]:''); stats.className='statline'; setTimeout(()=>pollIndex(slug),900); return; }
  if(a.error){ stats.textContent='Error al indexar: '+a.error; stats.className='statline err'; btnIndex.disabled=false; return; }
  // termino bien
  stats.textContent='Indice "'+(a.slug||slug||'')+'" listo. Activandolo...'; stats.className='statline ok';
  await refreshIndices(a.slug||slug);
  if(a.slug||slug) await cambiarIndice(a.slug||slug);
  btnIndex.disabled=false;
  if(a.slug||slug) stats.textContent='Indice "'+(a.slug||slug)+'" listo y activo.'; 
}
function busy(b){ send.disabled=b; inp.disabled=b; }
function autosize(){ inp.style.height='auto'; inp.style.height=Math.min(inp.scrollHeight,140)+'px'; }

selIndex.addEventListener('change',async()=>{ if(selIndex.value) await cambiarIndice(selIndex.value); });
btnRefresh.addEventListener('click',()=>refreshIndices());
btnIndex.addEventListener('click',doIndex);
selModel.addEventListener('change',applyModel);
btnModel.addEventListener('click',applyModel);
kInput.addEventListener('keydown',e=>{ if(e.key==='Enter'){ e.preventDefault(); applyK(); } });
btnK.addEventListener('click',applyK);
inp.addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); ask(); } });
inp.addEventListener('input',autosize);
send.addEventListener('click',ask);

(async function init(){
  const active=await jget('/api/active');
  addWelcome();
  await refreshModels();
  await refreshConfig();
  await refreshIndices(active.slug||undefined);
  if(active.slug){ slugActual=active.slug; cargarChat(active.slug); }
  await refreshActive();
  inp.focus();
})();
</script>
</body>
</html>"""


# ============================================================================
# API + manejo de peticiones
# ============================================================================
def _json(code, obj):
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return (code, body, "application/json; charset=utf-8")


def _html(body):
    return (200, body.encode("utf-8"), "text/html; charset=utf-8")


def _list_ollama_models():
    """Devuelve la lista de modelos disponibles en Ollama (config.OLLAMA_HOST),
    con el nombre, el tamano (parametros) y la familia, para mostrarlos
    ordenados por familia y tamano en la interfaz."""
    try:
        req = urllib.request.Request(
            f"http://{config.OLLAMA_HOST}/api/tags",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    out = []
    for m in data.get("models", []):
        det = m.get("details", {}) or {}
        q = det.get("quantization_level") or ""
        fam = (det.get("family") or "otro").lower()
        para = (det.get("parameter_size") or "?").upper().rstrip("B")
        try:
            size_gb = (m.get("size") or 0) / (1024 ** 3)
            gb = f"{size_gb:.1f}GB"
        except Exception:
            gb = "?"
        out.append({"name": m.get("name"), "family": fam,
                    "parametros": para, "gb": gb, "quant": q})
    out.sort(key=lambda x: (_fam_order(x["family"]), _size_num(x["parametros"])))
    return out


def _fam_order(fam):
    # mapas de familia de Ollama -> orden de aparicion del grupo en el selector
    first = ["gemma", "gemma2", "gemma3", "llama", "qwen", "qwen2", "qwen25vl",
             "qwen3", "granite", "phi", "phi2", "phi3", "deepseek", "ds", "moondream"]
    for i, f in enumerate(first):
        if fam == f or fam.startswith(f):
            return i
    return len(first)


def _size_num(s):
    try:
        return float(s)
    except Exception:
        return 999.0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _reply(self, resp):
        code, body, ctype = resp
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._reply(_html(HTML))
            elif path == "/api/indices":
                self._reply(_json(200, RIX.list_indices()))
            elif path == "/api/status":
                self._reply(_json(200, {
                    "active": _INDEXING["active"],
                    "slug": _INDEXING["slug"],
                    "done": _INDEXING["done"],
                    "error": _INDEXING["error"],
                    "log": _INDEXING["log"][-40:],
                }))
            elif path == "/api/active":
                a = get_active()
                self._reply(_json(200, {"slug": a["slug"] if a else None,
                                        "archivo_base": a["archivo_base"] if a else None,
                                        "path": a["path"] if a else None}
                                  if a else {"slug": None, "archivo_base": None}))
            elif path == "/api/models":
                self._reply(_json(200, {
                    "modelos": _list_ollama_models(),
                    "activo": config.MODEL,
                }))
            elif path == "/api/config":
                self._reply(_json(200, {"modelo": config.MODEL, "k": K}))
            else:
                self._reply(_json(404, {"error": "not found"}))
        except Exception as e:
            self._reply(_json(500, {"error": str(e)}))

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        payload = {}
        if raw:
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {}
        try:
            if path == "/api/ask":
                q = (payload.get("question") or "").strip()[:2000]
                if not q:
                    self._reply(_json(400, {"error": "pregunta vacia"}))
                    return
                a = get_active()
                if a is None:
                    _default_index()
                    a = get_active()
                if a is None:
                    self._reply(_json(400, {"error": "No hay indice activo. Pega una ruta y pulsa Indexar, o crea una con indexar_carpeta.bat (o --index)."}))
                    return
                res = _ask(q, a)
                self._reply(_json(200, res))
            elif path == "/api/index":
                folder = (payload.get("folder") or "").strip()
                if not folder or not os.path.isdir(folder):
                    self._reply(_json(400, {"error": "ruta no valida o no existe"}))
                    return
                slug = RIX.slugify(os.path.basename(os.path.normpath(folder)))
                do_index(folder)
                self._reply(_json(200, {"ok": True, "slug": slug}))
            elif path == "/api/select":
                idx = (payload.get("index") or "").strip()
                path_pkl = RIX.index_path_for(idx)
                if not os.path.isfile(path_pkl):
                    self._reply(_json(400, {"error": f"indice no existe: {idx}"}))
                    return
                _set_active(path_pkl, idx)
                self._reply(_json(200, {"ok": True, "slug": idx}))
            elif path == "/api/model":
                m = (payload.get("model") or "").strip()
                if not m:
                    self._reply(_json(400, {"error": "modelo vacio"}))
                    return
                config.set_model(m)
                self._reply(_json(200, {"ok": True, "activo": config.MODEL}))
            elif path == "/api/k":
                try:
                    k = set_k(int((payload.get("k") or K)))
                except Exception:
                    self._reply(_json(400, {"error": "k invalido"}))
                    return
                self._reply(_json(200, {"ok": True, "k": K}))
            else:
                self._reply(_json(404, {"error": "not found"}))
        except Exception as e:
            self._reply(_json(500, {"error": str(e)}))


def _ask(question, active):
    enc = _get_encoder()
    t0 = time.time()
    qemb = enc.encode([question], normalize_embeddings=True)[0]
    cands = RR.select_ranked(active["emb"], qemb, active["src"], active["chunks"], K, query=question)
    raw = active["emb"] @ qemb
    hits = RR.hit_tuple(cands, active["chunks"], active["src"], raw, query=question)
    prompt = RR.build_prompt(question, hits)
    res = runner.run_single(prompt)
    elapsed = time.time() - t0
    base = active.get("archivo_base", "")
    sources = []
    for h in hits:
        rel = h["ruta"]
        abs_path = os.path.normpath(os.path.join(base, rel)) if base else rel
        sources.append({
            "ruta": rel,
            "abs": abs_path,
            "coseno": round(float(h["score_bruto"]), 3),
            "ajuste": round(float(h["ajuste"]), 3),
            "texto": h["texto"],
        })
    return {
        "question": question,
        "answer": (res.get("response") or "").strip(),
        "sources": sources,
        "error": res.get("error"),
        "tiempo_s": round(elapsed, 2),
        "modelo": config.MODEL,
        "k": K,
    }


def main():
    import webbrowser
    os.makedirs(RIX.INDICES_DIR, exist_ok=True)
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}/"
    print(f"\nServidor RAG web corriendo en {url}")
    print("Pega la ruta de una carpeta, pulsa Indexar, elige el proyecto y pregunta.")
    print("Presiona Ctrl+C para detener.\n")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nDetenido.")
        srv.server_close()


if __name__ == "__main__":
    main()
