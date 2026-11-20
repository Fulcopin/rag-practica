"""
Sube todos los CVs de una carpeta al agente (POST /cvs/upload).

Uso:
    python subir_cvs.py                         # ./cvs_ejemplo contra 127.0.0.1:8000
    python subir_cvs.py C:\\mis_cvs --pausa 15   # tu carpeta, con pausa entre CVs
    python subir_cvs.py --url http://127.0.0.1:8000

La pausa existe por el limite de tokens por minuto de las APIs gratuitas
(Groq: ~8K tok/min). Cada CV gasta ~1.000 tokens al extraer su payload.
"""
import argparse
import os
import sys
import time

import httpx


def main() -> None:
    ap = argparse.ArgumentParser(description="Sube CVs al agente de Qdrant")
    ap.add_argument("carpeta", nargs="?", default="cvs_ejemplo")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--pausa", type=float, default=12,
                    help="segundos entre CVs (limite de tokens por minuto)")
    ap.add_argument("--reintentos", type=int, default=3)
    a = ap.parse_args()

    if not os.path.isdir(a.carpeta):
        sys.exit(f"No existe la carpeta: {a.carpeta}")
    archivos = sorted(f for f in os.listdir(a.carpeta)
                      if f.lower().endswith((".pdf", ".txt")))
    if not archivos:
        sys.exit(f"No hay archivos .pdf ni .txt en {a.carpeta}")

    print(f"{len(archivos)} CVs -> {a.url}  (pausa {a.pausa}s)\n")
    ok = 0
    for i, nombre in enumerate(archivos, 1):
        ruta = os.path.join(a.carpeta, nombre)
        tipo = "application/pdf" if nombre.lower().endswith(".pdf") else "text/plain"
        for intento in range(1, a.reintentos + 1):
            try:
                with open(ruta, "rb") as fh:
                    r = httpx.post(f"{a.url.rstrip('/')}/cvs/upload", timeout=300,
                                   files={"archivos": (nombre, fh.read(), tipo)})
            except httpx.HTTPError as e:
                print(f"[{i}/{len(archivos)}] {nombre}: error de red ({e})")
                time.sleep(5)
                continue
            if r.status_code == 429 and intento < a.reintentos:
                print(f"[{i}/{len(archivos)}] {nombre}: limite por minuto, espero 30s")
                time.sleep(30)
                continue
            if r.status_code != 200:
                print(f"[{i}/{len(archivos)}] {nombre}: HTTP {r.status_code} {r.text[:200]}")
                break
            datos = r.json()
            p = datos[next(iter(datos))]
            print(f"[{i}/{len(archivos)}] OK {nombre}: {p['habilidades']} "
                  f"| anios={p['anios']} | {p['idiomas']}")
            ok += 1
            break
        if i < len(archivos):
            time.sleep(a.pausa)

    print(f"\n{ok}/{len(archivos)} subidos. Comprueba con: GET {a.url.rstrip('/')}/cvs")
    sys.exit(0 if ok == len(archivos) else 1)


if __name__ == "__main__":
    main()
