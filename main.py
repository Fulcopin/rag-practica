"""
Agente de CVs: RAG por PAYLOAD + base de datos vectorial GRATIS (Qdrant) + Hugging Face API.

Qdrant se puede usar:
  - LOCAL, sin cuenta y sin internet (guarda en ./qdrant_data)        <- por defecto
  - Qdrant Cloud gratis (1 GB RAM / 4 GB disco): define QDRANT_URL y QDRANT_API_KEY

Ejecutar:
    pip install -r requirements.txt
    export HF_TOKEN="hf_xxx"        (o crea un archivo .env)
    uvicorn main:app --reload      ->  http://127.0.0.1:8000/docs

En Windows lo mas comodo es:  iniciar.ps1  (o  .\\iniciar.ps1  en PowerShell)
"""
import io
import json
import os
import re
import unicodedata
import uuid

import numpy as np
import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from huggingface_hub import InferenceClient
from pydantic import BaseModel
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _cargar_env(ruta: str) -> None:
    """Carga un .env simple (KEY=VALOR) sin dependencias extra. No pisa el entorno."""
    if not os.path.exists(ruta):
        return
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#") or "=" not in linea:
                continue
            k, v = linea.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_cargar_env(os.path.join(BASE_DIR, ".env"))

HF_TOKEN = os.getenv("HF_TOKEN")
# Backend de embeddings: "local" (fastembed, gratis) o "hf" (API de pago)
EMB_BACKEND = os.getenv("EMB_BACKEND", "hf" if HF_TOKEN else "local").lower()
# Backend del LLM: "ollama" (local, gratis) o "hf" (API de pago)
LLM_BACKEND = os.getenv("LLM_BACKEND", "hf" if HF_TOKEN else "ollama").lower()
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LOCAL_EMB_MODEL = os.getenv("LOCAL_EMB_MODEL", EMB_MODEL)
# Donde se guarda el modelo ONNX (240 MB). Junto al proyecto para que no se borre.
FASTEMBED_CACHE = os.getenv("FASTEMBED_CACHE", os.path.join(BASE_DIR, "models_cache"))
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
LLM_PROVIDER = os.getenv("LLM_PROVIDER") or None  # ej. "novita"; vacio = router automatico de HF
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
# Cualquier API compatible con OpenAI (Groq, Gemini, OpenRouter, LM Studio, Ollama /v1...)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen/qwen3.8-27b")
# Solo para modelos de razonamiento (gpt-oss): "low" | "medium" | "high". Vacio = no enviar.
OPENAI_REASONING = os.getenv("OPENAI_REASONING") or None
QDRANT_URL = os.getenv("QDRANT_URL")            # ej: https://xxxx.cloud.qdrant.io:6333
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLECCION = os.getenv("QDRANT_COLLECTION", "cvs")
QDRANT_PATH = os.getenv("QDRANT_PATH", os.path.join(BASE_DIR, "qdrant_data"))
if "hf" in (EMB_BACKEND, LLM_BACKEND) and not HF_TOKEN:
    raise RuntimeError(
        "Falta HF_TOKEN y algun backend usa Hugging Face.\n"
        "  1) Crea un token en https://huggingface.co/settings/tokens\n"
        "  2) PowerShell:  $env:HF_TOKEN = \"hf_xxx\"\n"
        "     CMD:         set HF_TOKEN=hf_xxx\n"
        "     o crea un archivo .env junto a main.py (iniciar.ps1 lo carga solo)\n"
        "  Sin token, el agente usa los backends locales (fastembed + Ollama)."
    )

client = InferenceClient(token=HF_TOKEN, provider=LLM_PROVIDER) if HF_TOKEN else None
# Qdrant local: la ruta es absoluta, asi funciona aunque lances uvicorn desde otra carpeta
qd = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_URL else QdrantClient(path=QDRANT_PATH)

# ==== NUCLEO ====
CATALOGO = ["python", "sql", "spark", "airflow", "aws", "azure", "gcp", "docker", "kubernetes",
            "fastapi", "django", "flask", "postgresql", "mongodb", "pytorch", "tensorflow",
            "nlp", "hugging face", "rag", "llm", "faiss", "langchain", "machine learning",
            "pandas", "etl", "java", "javascript", "react", "git", "figma", "photoshop"]
ALIAS = {"huggingface": "hugging face", "hf": "hugging face", "ml": "machine learning",
         "postgres": "postgresql", "llms": "llm", "js": "javascript", "k8s": "kubernetes"}
IDIOMAS = ["ingles", "espanol", "frances", "portugues", "aleman", "italiano"]
# Un CV en ingles devuelve "english"/"spanish": lo unificamos al nombre en espanol.
ALIAS_IDIOMA = {"english": "ingles", "spanish": "espanol", "french": "frances",
                "portuguese": "portugues", "german": "aleman", "italian": "italiano",
                "castellano": "espanol"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(c for c in s if not unicodedata.combining(c)).strip()


def canon(t: str) -> str:
    t = norm(t)
    return ALIAS.get(t, t)


def canon_idioma(t: str) -> str:
    t = norm(t)
    return ALIAS_IDIOMA.get(t, t)


def _palabra(term: str, texto: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", texto) is not None


def payload_por_reglas(texto: str) -> dict:
    """Extracción rápida sin LLM: catálogo + regex."""
    t = norm(texto)
    hab = {s for s in CATALOGO if _palabra(s, t)}
    hab |= {c for a, c in ALIAS.items() if _palabra(a, t)}
    anios = [int(x) for x in re.findall(r"(\d+)\s*\+?\s*anos", t)]
    idiomas = {i for i in IDIOMAS if i in t} | {c for a, c in ALIAS_IDIOMA.items() if _palabra(a, t)}
    return {"habilidades": sorted(hab), "anios": max(anios) if anios else None,
            "idiomas": sorted(idiomas)}


def llm(system: str, user: str, max_tokens: int = 600) -> str:
    """Respuesta del LLM: Ollama local, API compatible con OpenAI, o Hugging Face."""
    if LLM_BACKEND == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("Falta OPENAI_API_KEY para el backend 'openai' "
                               "(Groq/Gemini/OpenRouter...): ponla en .env")
        try:
            cuerpo = {"model": OPENAI_MODEL, "temperature": 0.1, "max_tokens": max_tokens,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]}
            if OPENAI_REASONING:
                cuerpo["reasoning_effort"] = OPENAI_REASONING
            r = httpx.post(f"{OPENAI_BASE_URL}/chat/completions", timeout=300,
                           headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}, json=cuerpo)
        except httpx.HTTPError as e:
            raise RuntimeError(f"No puedo conectar con {OPENAI_BASE_URL}. ({e})") from e
        if r.status_code in (401, 403):
            raise RuntimeError(f"API key rechazada por {OPENAI_BASE_URL} (HTTP {r.status_code}). Revisa OPENAI_API_KEY")
        if r.status_code == 429:
            raise RuntimeError(f"Limite de peticiones de {OPENAI_MODEL} alcanzado (HTTP 429). Espera un poco")
        if r.status_code == 404:
            raise RuntimeError(f"El modelo '{OPENAI_MODEL}' no existe en {OPENAI_BASE_URL}. Revisa OPENAI_MODEL")
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    if LLM_BACKEND == "ollama":
        try:
            r = httpx.post(f"{OLLAMA_URL}/api/chat", timeout=300, json={
                "model": OLLAMA_MODEL, "stream": False,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "options": {"temperature": 0.1, "num_predict": max_tokens}})
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama no responde en {OLLAMA_URL}. Abre una terminal y ejecuta "
                               f"'ollama serve' (o la app de Ollama) y 'ollama pull {OLLAMA_MODEL}'. ({e})") from e
        if r.status_code == 404:
            raise RuntimeError(f"Ollama no tiene el modelo '{OLLAMA_MODEL}'. Ejecuta: ollama pull {OLLAMA_MODEL}")
        if r.status_code == 500 and "memory" in r.text.lower():
            raise RuntimeError("Ollama no tiene RAM suficiente para el modelo. Cierra programas "
                               f"(Chrome) o usa un modelo mas pequeno: ollama pull {OLLAMA_MODEL}")
        r.raise_for_status()
        return r.json()["message"]["content"]
    r = client.chat_completion(model=LLM_MODEL, max_tokens=max_tokens, temperature=0.1,
                               messages=[{"role": "system", "content": system},
                                         {"role": "user", "content": user}])
    return r.choices[0].message.content


def _json(txt: str) -> dict:
    try:
        return json.loads(re.search(r"\{.*\}", txt, re.S).group())
    except Exception:
        return {}


def payload_cv(texto: str) -> dict:
    """Payload del CV = LLM (estructura) + reglas (red de seguridad)."""
    base = payload_por_reglas(texto)
    d = _json(llm('Extrae del CV SOLO JSON: {"habilidades":[...],"anios":int|null,"idiomas":[...]}. '
                  "Habilidades = tecnologías y herramientas, en minúscula.", texto[:3500], 300))
    hab = {canon(h) for h in d.get("habilidades", []) if isinstance(h, str)} | set(base["habilidades"])
    anios = d.get("anios") if isinstance(d.get("anios"), int) else base["anios"]
    idi = {canon_idioma(i) for i in d.get("idiomas", []) if isinstance(i, str)} | set(base["idiomas"])
    return {"habilidades": sorted(hab), "anios": anios, "idiomas": sorted(idi)}


def payload_pregunta(pregunta: str) -> dict:
    """Convierte la pregunta en diccionario de términos clave."""
    d = _json(llm('Convierte la pregunta de un reclutador en SOLO JSON: '
                  '{"habilidades":[...],"anios_min":int|null,"idiomas":[...]}. '
                  "Solo términos clave, en minúscula, sin frases.", pregunta, 200))
    base = payload_por_reglas(pregunta)
    hab = {canon(h) for h in d.get("habilidades", []) if isinstance(h, str)} | set(base["habilidades"])
    anios = d.get("anios_min") if isinstance(d.get("anios_min"), int) else base["anios"]
    idi = {canon_idioma(i) for i in d.get("idiomas", []) if isinstance(i, str)} | set(base["idiomas"])
    return {"habilidades": sorted(hab), "anios_min": anios, "idiomas": sorted(idi)}


_emb_local = None


def _modelo_embeddings_local():
    """Carga perezosa: fastembed descarga el modelo ONNX la primera vez (~470 MB)."""
    global _emb_local
    if _emb_local is None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise RuntimeError("Falta fastembed para embeddings locales: "
                               "pip install -r requirements.txt") from e
        _emb_local = TextEmbedding(model_name=LOCAL_EMB_MODEL, cache_dir=FASTEMBED_CACHE)
    return _emb_local


def embed(texto: str) -> np.ndarray:
    """Vector normalizado: fastembed local (gratis) o Hugging Face (de pago)."""
    if EMB_BACKEND == "local":
        v = np.array(next(_modelo_embeddings_local().embed([texto])), dtype=np.float32)
    else:
        v = np.array(client.feature_extraction(texto, model=EMB_MODEL), dtype=np.float32)
    if v.ndim > 1:
        v = v.mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def _match(req: str, skills: list[str]) -> bool:
    return any(req == s or (len(req) > 2 and (req in s or s in req)) for s in skills)


def puntuar(req: dict, cand: dict) -> tuple[float, list[str]]:
    """Compara diccionario vs diccionario. Devuelve (score 0-1, habilidades faltantes)."""
    partes, faltan = [], []
    reqs = req.get("habilidades") or []
    if reqs:
        faltan = [r for r in reqs if not _match(r, cand["habilidades"])]
        partes.append((0.7, (len(reqs) - len(faltan)) / len(reqs)))
    if req.get("anios_min"):
        partes.append((0.2, min(1.0, (cand["anios"] or 0) / req["anios_min"])))
    if req.get("idiomas"):
        partes.append((0.1, sum(i in cand["idiomas"] for i in req["idiomas"]) / len(req["idiomas"])))
    if not partes:
        return 0.0, faltan
    return sum(w * v for w, v in partes) / sum(w for w, _ in partes), faltan


# ---------- Base de datos vectorial: Qdrant ----------
# Lección del repo de Azure AI Search: las frases comunes ("Hoja de vida") ensucian el embedding.
# Por eso se limpian y se antepone nombre + habilidades (equivale a title + keywords del índice de Azure).
BOILERPLATE = re.compile(r"\b(hoja de vida|curriculum vitae|curr[ií]culum|cv)\b\s*[:\-]?", re.I)


def texto_para_embedding(nombre: str, p: dict, texto: str) -> str:
    return f"{nombre.replace('_', ' ')}. Habilidades: {', '.join(p['habilidades'])}. {BOILERPLATE.sub('', texto)[:1500]}"


def asegurar_coleccion(dim: int) -> None:
    if qd.collection_exists(COLECCION):
        return
    qd.create_collection(COLECCION, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
    if QDRANT_URL:  # en Qdrant Cloud conviene indexar los campos del payload (en modo local no hace falta)
        qd.create_payload_index(COLECCION, "habilidades", models.PayloadSchemaType.KEYWORD)
        qd.create_payload_index(COLECCION, "anios", models.PayloadSchemaType.INTEGER)


def indexar_cv(nombre: str, texto: str) -> dict:
    p = payload_cv(texto)
    vec = embed(texto_para_embedding(nombre, p, texto)).tolist()
    asegurar_coleccion(len(vec))
    qd.upsert(COLECCION, points=[models.PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, nombre)),   # mismo nombre = se actualiza, no se duplica
        vector=vec,
        payload={"candidato": nombre, **p, "texto": texto})])
    return p


def evidencia(texto: str, req: dict, n: int = 2) -> list[str]:
    frases = [f.strip() for f in re.split(r"(?<=[.!?\n])\s+", texto) if f.strip()]
    terms = req.get("habilidades") or []
    return sorted(frases, key=lambda f: -sum(_palabra(t, norm(f)) for t in terms))[:n]


def buscar_por_payload(req: dict, k: int = 5) -> list[dict]:
    """1) Qdrant filtra por payload (habilidades) y ordena por vector  2) se re-puntúa con el diccionario."""
    if not qd.collection_exists(COLECCION):
        raise ValueError("Primero sube CVs")
    hab = req.get("habilidades") or []
    if not hab:
        raise ValueError("El payload necesita al menos una habilidad en 'habilidades'")
    qvec = embed("Habilidades: " + ", ".join(hab)).tolist()
    filtro = models.Filter(must=[models.FieldCondition(key="habilidades", match=models.MatchAny(any=hab))])
    pts = qd.query_points(COLECCION, query=qvec, query_filter=filtro, limit=k * 5, with_payload=True).points
    if not pts:  # nadie cumple ni una habilidad: caemos a búsqueda puramente semántica
        pts = qd.query_points(COLECCION, query=qvec, limit=k * 5, with_payload=True).points
    res = []
    for pt in pts:
        pl = pt.payload
        score, faltan = puntuar(req, pl)
        res.append({"candidato": pl["candidato"], "score": round(0.75 * score + 0.25 * pt.score, 3),
                    "faltan": faltan,
                    "payload": {c: pl.get(c) for c in ("habilidades", "anios", "idiomas")},
                    "evidencia": evidencia(pl["texto"], req)})
    return sorted(res, key=lambda r: -r["score"])[:k]


def listar_candidatos() -> dict:
    if not qd.collection_exists(COLECCION):
        return {}
    pts, _ = qd.scroll(COLECCION, limit=500, with_payload=["candidato", "habilidades", "anios", "idiomas"])
    return {p.payload["candidato"]: {c: p.payload.get(c) for c in ("habilidades", "anios", "idiomas")} for p in pts}


def borrar_todo() -> None:
    if qd.collection_exists(COLECCION):
        qd.delete_collection(COLECCION)


def agente(pregunta: str, k: int = 3) -> dict:
    req = payload_pregunta(pregunta)                 # 1) pregunta -> payload
    ranking = buscar_por_payload(req, k)             # 2) Qdrant (filtro + vector) + puntuación por payload
    ctx = "\n\n".join(f"[{r['candidato']}] score={r['score']} faltan={r['faltan']}\n"
                      f"payload={r['payload']}\nEvidencia: {' | '.join(r['evidencia'])}" for r in ranking)
    respuesta = llm("Eres un asistente de selección. Usa SOLO los datos dados, justifica el ranking "
                    "con evidencia y menciona lo que falta. No inventes. Español.",
                    f"Pregunta: {pregunta}\nPayload buscado: {req}\n\nTop candidatos:\n{ctx}")  # 3) explicar
    return {"payload": req, "ranking": ranking, "respuesta": respuesta}
# ==== FIN NUCLEO ====


def leer_archivo(nombre: str, contenido: bytes) -> str:
    if nombre.lower().endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(contenido)).pages)
    return contenido.decode("utf-8", errors="ignore")


# ---------------- API ----------------
app = FastAPI(title="Agente RAG de CVs (payload + Qdrant)", version="3.0")


@app.get("/panel", include_in_schema=False)
def panel():
    """Panel web: subir CVs y preguntar sin usar /docs."""
    ruta = os.path.join(BASE_DIR, "panel.html")
    if not os.path.exists(ruta):
        raise HTTPException(404, "Falta el archivo panel.html junto a main.py")
    return FileResponse(ruta, media_type="text/html; charset=utf-8")


@app.get("/", include_in_schema=False)
def inicio():
    """Portada simple con enlaces utiles (antes daba 404)."""
    try:
        total = len(listar_candidatos())
    except Exception:
        total = 0
    qd_modo = "Qdrant Cloud" if QDRANT_URL else "Qdrant local"
    return HTMLResponse(f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agente de CVs</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, Segoe UI, sans-serif; max-width: 720px;
         margin: 6vh auto; padding: 0 20px; line-height: 1.6; }}
  h1 {{ font-size: 1.7rem; margin-bottom: .2rem; }}
  .sub {{ opacity: .7; margin-top: 0; }}
  ul {{ padding-left: 1.2rem; }}
  li {{ margin: .45rem 0; }}
  code {{ background: rgba(127,127,127,.15); padding: .1rem .35rem; border-radius: 4px; }}
  .estado {{ margin-top: 1.5rem; padding: .8rem 1rem; border-radius: 8px;
             background: rgba(127,127,127,.12); font-size: .93rem; }}
</style></head>
<body>
  <h1>Agente de CVs</h1>
  <p class="sub">RAG por payload + Qdrant + embeddings locales</p>
  <ul>
    <li><a href="/panel">Panel web</a> — subir CVs y preguntar con botones (<code>/panel</code>)</li>
    <li><a href="/docs">Documentación interactiva</a> — probar todo desde el navegador (<code>/docs</code>)</li>
    <li><a href="/cvs">Candidatos indexados</a> — <code>GET /cvs</code></li>
    <li><a href="/salud">Diagnóstico</a> — modelos, backend y conexión (<code>/salud</code>)</li>
    <li><code>POST /cvs/upload</code> · <code>POST /search</code> · <code>POST /agent</code></li>
  </ul>
  <div class="estado">
    <b>{total}</b> candidatos · Embeddings: <code>{EMB_BACKEND}</code> ·
    LLM: <code>{LLM_BACKEND}</code> · {qd_modo}
  </div>
</body></html>""")


class Pregunta(BaseModel):
    pregunta: str
    k: int = 3


class BusquedaPayload(BaseModel):
    # Ejemplo: {"habilidades": ["python","rag"], "anios_min": 3, "idiomas": ["ingles"]}
    payload: dict
    k: int = 5


@app.post("/cvs/upload")
async def subir_cvs(archivos: list[UploadFile] = File(...)):
    """Sube CVs (.pdf/.txt): extrae payload, genera embedding y guarda en Qdrant."""
    out = {}
    for f in archivos:
        nombre = os.path.splitext(f.filename)[0]
        out[nombre] = indexar_cv(nombre, leer_archivo(f.filename, await f.read()))
    return out


@app.get("/cvs")
def listar():
    return listar_candidatos()


@app.delete("/cvs")
def reiniciar():
    borrar_todo()
    return {"ok": True}


@app.post("/payload")
def ver_payload(p: Pregunta):
    return payload_pregunta(p.pregunta)


@app.post("/search")
def buscar(b: BusquedaPayload):
    """Búsqueda con tu propio diccionario (sin LLM)."""
    try:
        return buscar_por_payload(b.payload, b.k)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/agent")
def preguntar(p: Pregunta):
    try:
        return agente(p.pregunta, p.k)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/salud")
def salud():
    """Diagnostico rapido: que backend se esta usando y si responde."""
    info = {
        "embeddings": {"backend": EMB_BACKEND,
                       "modelo": LOCAL_EMB_MODEL if EMB_BACKEND == "local" else EMB_MODEL},
        "llm": {"backend": LLM_BACKEND,
                "modelo": {"ollama": OLLAMA_MODEL, "openai": OPENAI_MODEL}.get(LLM_BACKEND, LLM_MODEL)},
        "qdrant": {"modo": "cloud" if QDRANT_URL else "local",
                   "coleccion": COLECCION,
                   "existe": qd.collection_exists(COLECCION)},
    }
    if LLM_BACKEND == "ollama":
        try:
            r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            info["llm"]["ollama"] = "ok" if r.status_code == 200 else f"http {r.status_code}"
            if r.status_code == 200:
                modelos = [m.get("name", "") for m in r.json().get("models", [])]
                info["llm"]["modelos"] = modelos
                info["llm"]["modelo_listo"] = any(m.split(":")[0] == OLLAMA_MODEL.split(":")[0]
                                                  for m in modelos)
        except httpx.HTTPError:
            info["llm"]["ollama"] = "no responde (ejecuta 'ollama serve')"
    return info


if __name__ == "__main__":
    import uvicorn

    puerto = int(os.getenv("PORT", "8000"))
    print(f"Agente de CVs en http://127.0.0.1:{puerto}/docs  (Ctrl+C para parar)")
    print(f"Embeddings: {EMB_BACKEND} | LLM: {LLM_BACKEND} | "
          f"Qdrant: {'cloud' if QDRANT_URL else 'local'}")
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=puerto)
