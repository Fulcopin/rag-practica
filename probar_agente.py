"""
Prueba local del agente SIN gastar tokens de Hugging Face.

Sirve para comprobar que Qdrant local, la extraccion de payload, el ranking
y la API (FastAPI) funcionan en esta computadora antes de usar la nube.

Uso:
    python probar_agente.py          # modo offline (embeddings simulados)
    python probar_agente.py --real   # usa Hugging Face de verdad (necesita HF_TOKEN)

En modo offline NO se llama a internet: se sustituyen `embed` y `llm` por
versiones deterministas. La base Qdrant se crea en una carpeta temporal, asi
que no toca tu ./qdrant_data real.
"""
import os
import sys
import tempfile
import re

REAL = "--real" in sys.argv
if not REAL:
    # Offline: nada de red. Forzamos backends locales y quitamos cualquier token.
    os.environ.setdefault("EMB_BACKEND", "local")
    os.environ.setdefault("LLM_BACKEND", "ollama")
    os.environ.pop("HF_TOKEN", None)

# Aislamos la base ANTES de importar main: ruta temporal, coleccion de prueba y
# sin Qdrant Cloud (para que la prueba nunca toque tus CVs reales ni tu nube).
_tmp = tempfile.mkdtemp(prefix="prueba_qdrant_")
os.environ["QDRANT_PATH"] = os.path.join(_tmp, "qdrant_data")
os.environ["QDRANT_COLLECTION"] = "cvs_prueba"
os.environ.pop("QDRANT_URL", None)
os.environ.pop("QDRANT_API_KEY", None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main  # noqa: E402

if not REAL:
    VOCAB = sorted(set(main.CATALOGO) | set(main.IDIOMAS) |
                   {"anos", "experiencia", "senior", "junior"})

    def embed_falso(texto: str):
        import numpy as np
        v = np.zeros(len(VOCAB), dtype=np.float32)
        t = main.norm(texto)
        for i, w in enumerate(VOCAB):
            if re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", t):
                v[i] = 1.0
        v[-1] = 0.1  # evita vector nulo
        return v / (np.linalg.norm(v) + 1e-9)

    def llm_falso(system: str, user: str, max_tokens: int = 600) -> str:
        return "Respuesta simulada: ranking basado en el payload y la evidencia."

    main.embed = embed_falso      # type: ignore[assignment]
    main.llm = llm_falso          # type: ignore[assignment]

CVS = {
    "ana_torres": "Ana Torres. Hoja de vida. 5 años de experiencia. Python, SQL, pandas y ETL. "
                  "Construi un RAG con LangChain y Qdrant. Ingles avanzado.",
    "luis_gomez": "Luis Gomez. Curriculum Vitae. 2 años como backend. Python, FastAPI, Docker, "
                  "PostgreSQL. Espanol nativo, ingles intermedio.",
    "marta_ruiz": "Marta Ruiz. 8 años en datos. Spark, Airflow, AWS, Kubernetes, Python. "
                  "Machine Learning con PyTorch. Frances e ingles.",
}

fallos = []


def check(nombre: str, cond: bool, extra: str = "") -> None:
    print(f"{'OK  ' if cond else 'FALLA'} {nombre}{(' -> ' + extra) if extra and not cond else ''}")
    if not cond:
        fallos.append(nombre)


print(f"\n== 1. Indexar 3 CVs en Qdrant local ({_tmp}) ==")
for nombre, texto in CVS.items():
    pl = main.indexar_cv(nombre, texto)
    check(f"indexado {nombre}", bool(pl["habilidades"]), str(pl))

print("\n== 2. Listar candidatos ==")
lista = main.listar_candidatos()
check("hay 3 candidatos", len(lista) == 3, str(sorted(lista)))

print("\n== 3. Busqueda por payload (sin LLM) ==")
req = {"habilidades": ["python", "rag"], "anios_min": 3, "idiomas": ["ingles"]}
res = main.buscar_por_payload(req, k=3)
for r in res:
    print(f"   {r['score']:.3f}  {r['candidato']}  faltan={r['faltan']}")
check("devuelve resultados", len(res) > 0)
check("ana_torres primera", res[0]["candidato"] == "ana_torres", str([r["candidato"] for r in res]))

print("\n== 4. Agente completo ==")
r = main.agente("Necesito un perfil de python con RAG y al menos 3 años", k=2)
check("el agente responde", bool(r["respuesta"]) and len(r["ranking"]) > 0)

print("\n== 5. API HTTP (TestClient: subir, listar, buscar, agente) ==")
from fastapi.testclient import TestClient  # noqa: E402

main.borrar_todo()
with TestClient(main.app) as c:
    archivos = [("archivos", (f"{n}.txt", t.encode("utf-8"), "text/plain")) for n, t in CVS.items()]
    rr = c.post("/cvs/upload", files=archivos)
    check("POST /cvs/upload 200", rr.status_code == 200, rr.text[:300])

    rr = c.get("/cvs")
    check("GET /cvs devuelve 3", rr.status_code == 200 and len(rr.json()) == 3, rr.text[:300])

    rr = c.post("/search", json={"payload": req, "k": 2})
    check("POST /search 200", rr.status_code == 200 and len(rr.json()) > 0, rr.text[:300])

    rr = c.post("/agent", json={"pregunta": "python con RAG", "k": 2})
    check("POST /agent 200", rr.status_code == 200 and "respuesta" in rr.json(), rr.text[:300])

    rr = c.post("/payload", json={"pregunta": "python senior ingles", "k": 1})
    check("POST /payload 200", rr.status_code == 200 and "python" in rr.json().get("habilidades", []), rr.text[:300])

    rr = c.delete("/cvs")
    check("DELETE /cvs 200", rr.status_code == 200, rr.text[:300])
    check("GET /cvs vacio tras borrar", c.get("/cvs").json() == {})

# Cerramos Qdrant a mano: si no, al salir imprime un ImportError inofensivo
try:
    main.qd.close()
except Exception:
    pass

print("\n" + ("TODO OK" if not fallos else f"FALLARON: {', '.join(fallos)}"))
sys.exit(1 if fallos else 0)
