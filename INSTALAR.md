# Agente de CVs — Qdrant + backends gratuitos

Agente que indexa CVs en **Qdrant** (base de datos vectorial) y responde
preguntas de reclutamiento con ranking + evidencia, sobre FastAPI.

Todo funciona **sin pagar**: embeddings en tu PC y LLM con una API gratuita
(Groq/Gemini) o local (Ollama).

| Pieza | Para qué |
|---|---|
| `main.py` | Lógica y API (FastAPI) |
| `probar_agente.py` | Autoprueba sin red (`--real` usa los backends reales) |
| `ver_qdrant.py` | Comprueba la conexión y muestra lo que hay en Qdrant |
| `subir_cvs.py` | Sube una carpeta entera de CVs con control del límite de tokens |
| `cvs_ejemplo/` | 8 CVs de ejemplo para probar sin datos reales |
| `iniciar.ps1` | Crea el entorno, carga `.env` y arranca el servidor |
| `empaquetar.ps1` | Genera el ZIP para llevarlo a otra PC |
| `requirements.txt` | Dependencias (probadas en Python 3.12 y 3.14) |
| `.env.example` | Plantilla de configuración |
| `models_cache/` | Se crea sola: modelo de embeddings ONNX (240 MB) |
| `qdrant_data/` | Se crea sola si usas Qdrant local |

---

## 1. Instalar en una PC nueva (Python 3.14.4)

Requisitos: Windows x64 y Python 3.14 (https://www.python.org/downloads/ —
marca *"Add python.exe to PATH"*).

```powershell
cd C:\ruta\al\proyecto
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env          # pega tu OPENAI_API_KEY (Groq o Gemini)
.\iniciar.ps1
```

Abre **http://127.0.0.1:8000/panel** (interfaz con botones: subir CVs y
preguntar), **/docs** para la API o `/salud` para el diagnóstico.

> Si PowerShell bloquea el script:
> `powershell -ExecutionPolicy Bypass -File .\iniciar.ps1`

`iniciar.ps1` ya crea el `.venv` e instala todo si no existe.
Con `-Reinstalar` rehace el entorno.

---

## 2. Elegir el LLM (los tres son gratis)

| Opción | Modelo | Ventajas | Límites aprox. |
|---|---|---|---|
| **Groq** (recomendado) | `qwen/qwen3.8-27b` | 1–3 s por respuesta, buen español, JSON limpio | 30 pet./min, 1.000/día, 8K tok/min |
| Groq | `openai/gpt-oss-120b` | Más potente, razona | mismo cupo; gasta tokens razonando (usa `OPENAI_REASONING=low`) |
| **Gemini** | `gemini-2.5-flash` | Cupo diario muy amplio | ~10–15 pet./min |
| **Ollama** | `gemma2:2b` | 100% offline, sin cuentas | necesita ~4 GB de RAM libre |

`.env` para Groq:

```ini
LLM_BACKEND=openai
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=gsk_...
OPENAI_MODEL=qwen/qwen3.8-27b
```

Para Gemini cambia tres líneas:

```ini
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
OPENAI_API_KEY=AIza...
OPENAI_MODEL=gemini-2.5-flash
```

Para Ollama (si tienes RAM):

```ini
LLM_BACKEND=ollama
OLLAMA_MODEL=gemma2:2b
```

> Hugging Face ya **no** da créditos gratis para Inference Providers (devuelve
> `402`). Solo se usa si pones `LLM_BACKEND=hf` con créditos/PRO, o si eliges el
> backend `hf` de embeddings. El paquete `huggingface_hub` sigue instalado
> porque `fastembed` lo usa **solo para descargar** el modelo una vez.

---

## 3. Embeddings: locales y gratis

```ini
EMB_BACKEND=local
LOCAL_EMB_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

- Corre con `fastembed` + `onnxruntime` en CPU (0,2 s por texto, 384 dims).
- La primera vez descarga 240 MB a `models_cache/` (sin token ni cuenta).
- Después funciona sin internet.
- Es el **mismo modelo** que usaba la API, así que los vectores son compatibles.

---

## 4. Probar el agente

```powershell
# a) Offline: sin red ni tokens, no toca tus datos
.\.venv\Scripts\python.exe probar_agente.py

# b) Real: embeddings locales + tu LLM, contra Qdrant (usa coleccion cvs_prueba)
.\.venv\Scripts\python.exe probar_agente.py --real

# c) Manual, en el navegador
.\iniciar.ps1       # -> /panel
```

**/panel** es lo más fácil: eliges archivos → *Subir*, escribes la pregunta y
pulsas *Solo ranking (sin LLM)* (1 llamada al LLM para extraer el payload) o
*Preguntar al agente* (ranking + redacción). Muestra puntuación, skills que
faltan y evidencia.

En `/docs` el orden útil: `POST /cvs/upload` → `GET /cvs` → `POST /payload` →
`POST /search` (sin LLM) → `POST /agent`.

### CVs de ejemplo

`cvs_ejemplo/` trae 11 perfiles variados (data engineering, backend, ML, NLP,
DevOps, BI, frontend, análisis, junior, data scientist y un CV en inglés). Con
el servidor arrancado:

```powershell
# sube la carpeta con pausas para no pasar el limite de tokens/minuto
.\.venv\Scripts\python.exe subir_cvs.py

# o tu propia carpeta de CVs
.\.venv\Scripts\python.exe subir_cvs.py C:\Users\tu_usuario\Documentos\cvs --pausa 15
```

El script imprime el payload extraído de cada CV. Para borrar los datos de
prueba: `Invoke-RestMethod http://127.0.0.1:8000/cvs -Method Delete`.

---

## 5. Qdrant: local o en la nube

### Local (por defecto) — gratis, sin cuenta

No definas `QDRANT_URL`. Los datos viven en `qdrant_data/` (ruta absoluta junto
a `main.py`; se cambia con `QDRANT_PATH`).

- Solo **un proceso** puede abrir esa carpeta a la vez.
- Para respaldar o migrar: copia la carpeta `qdrant_data/` completa.

### Nube (Qdrant Cloud, 1 GB gratis)

```powershell
QDRANT_URL=https://xxxx.cloud.qdrant.io:6333
QDRANT_API_KEY=tu_api_key
```

En la nube el código crea además índices de payload (`habilidades`, `anios`),
lo que acelera los filtros con muchos CVs.

Comprobar la conexión:

```powershell
.\.venv\Scripts\python.exe ver_qdrant.py
.\.venv\Scripts\python.exe ver_qdrant.py --probar   # escribe y borra una coleccion temporal
```

---

## 6. Llevarlo a otra computadora

```powershell
.\empaquetar.ps1                    # sin datos
.\empaquetar.ps1 -ConDatos          # + qdrant_data (CVs indexados en local)
.\empaquetar.ps1 -ConModelos        # + models_cache (evita los 240 MB de descarga)
```

En la PC nueva: descomprime y sigue el punto 1. **No copies `.venv/`** (no es
portable entre PCs).

Con Qdrant Cloud no hace falta copiar `qdrant_data/`: basta el mismo
`QDRANT_URL` + `QDRANT_API_KEY` en el `.env` de la otra PC.

> El `.env` contiene claves: no lo subas a Git ni lo compartas.

---

## 7. Ejemplos con PowerShell

```powershell
$form = @{ archivos = Get-Item .\cvs\ana_torres.pdf, .\cvs\luis_gomez.txt }
Invoke-RestMethod http://127.0.0.1:8000/cvs/upload -Method Post -Form $form

Invoke-RestMethod http://127.0.0.1:8000/cvs

Invoke-RestMethod http://127.0.0.1:8000/search -Method Post -ContentType "application/json" `
  -Body '{"payload":{"habilidades":["python","qdrant"],"anios_min":2,"idiomas":["ingles"]},"k":3}'

Invoke-RestMethod http://127.0.0.1:8000/agent -Method Post -ContentType "application/json" `
  -Body '{"pregunta":"Necesito un perfil de Python con RAG y al menos 3 años","k":3}'

Invoke-RestMethod http://127.0.0.1:8000/salud

Invoke-RestMethod http://127.0.0.1:8000/cvs -Method Delete
```

---

## 8. Problemas comunes

| Síntoma | Causa / solución |
|---|---|
| `402 Payment Required` (Hugging Face) | Sin créditos. Usa Groq/Gemini/Ollama (`LLM_BACKEND=openai/ollama`). |
| `401` / `403` en Groq-Gemini | `OPENAI_API_KEY` mal copiada o revocada. |
| `429 Too Many Requests` | Límite por minuto (Groq 8K tokens/min). Espera y reintenta. |
| `model ... not found` | El modelo no existe en ese proveedor: revisa `OPENAI_MODEL`. |
| El JSON del LLM se corta | `max_tokens` bajo o modelo de razonamiento (usa `OPENAI_REASONING=low`). |
| Ollama: `requires more system memory` | Cierra Chrome/apps o usa un modelo menor (`gemma2:2b`, `qwen2.5:0.5b`). |
| `already accessed by another instance` | Dos procesos abren `qdrant_data`. Cierra el otro. |
| `422` al subir | El campo debe llamarse `archivos` y ser uno o más ficheros. |
| Acentos raros en la consola | Es la codificación de PowerShell; el JSON de la API está bien (UTF-8). |
| PDF sin texto | Es un escaneo (imagen): necesita OCR antes. |

---

## 9. Qué hace por dentro

1. `payload_cv`: LLM + reglas → `{habilidades, anios, idiomas}`.
2. `texto_para_embedding`: nombre + habilidades + texto sin "hoja de vida"/"CV",
   para que el embedding no se contamine con frases genéricas.
3. `indexar_cv`: vector + payload en Qdrant (mismo nombre ⇒ actualiza, no duplica).
4. `buscar_por_payload`: filtro por `habilidades` → vector → re-puntuación
   ponderada (0,75 diccionario + 0,25 similitud) → evidencia textual.
5. `agente`: explicación final del ranking con el LLM, solo con los datos dados.
