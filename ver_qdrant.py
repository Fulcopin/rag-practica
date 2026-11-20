"""
Verifica la conexion a Qdrant (local o Cloud) y muestra lo que hay dentro.

Uso:
    python ver_qdrant.py             # solo lectura
    python ver_qdrant.py --probar    # ademas escribe y borra una coleccion de prueba

Lee la configuracion de .env o del entorno:
    QDRANT_URL, QDRANT_API_KEY   -> si hay URL usa Cloud; si no, local
    QDRANT_COLLECTION, QDRANT_PATH
"""
import os
import sys

from qdrant_client import QdrantClient, models

RAIZ = os.path.dirname(os.path.abspath(__file__))


def cargar_env():
    ruta = os.path.join(RAIZ, ".env")
    if not os.path.exists(ruta):
        return
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#") or "=" not in linea:
                continue
            k, v = linea.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


cargar_env()

URL = os.getenv("QDRANT_URL")
KEY = os.getenv("QDRANT_API_KEY")
COL = os.getenv("QDRANT_COLLECTION", "cvs")
PATH = os.getenv("QDRANT_PATH", os.path.join(RAIZ, "qdrant_data"))


def main():
    if URL:
        print(f"Modo:       CLOUD -> {URL.rstrip('/')}")
    else:
        print(f"Modo:       LOCAL -> {PATH}")
    qd = QdrantClient(url=URL.rstrip("/"), api_key=KEY, timeout=30) if URL else QdrantClient(path=PATH)

    try:
        nombres = [c.name for c in qd.get_collections().collections]
        print(f"Colecciones: {nombres or '(ninguna)'}")

        for n in nombres:
            d = qd.get_collection(n)
            print(f"  - {n}: {d.points_count} puntos, dim={d.config.params.vectors.size}, "
                  f"distancia={d.config.params.vectors.distance}")
            if d.payload_schema:
                print(f"    indices de payload: {sorted(d.payload_schema)}")

        if COL in nombres:
            pts, _ = qd.scroll(COL, limit=5, with_payload=["candidato", "habilidades", "anios", "idiomas"])
            print(f"\nPrimeros {len(pts)} de '{COL}':")
            for p in pts:
                pl = p.payload or {}
                print(f"  * {pl.get('candidato', '?')}: habilidades={pl.get('habilidades')} "
                      f"anios={pl.get('anios')} idiomas={pl.get('idiomas')}")
        else:
            print(f"\nLa coleccion '{COL}' aun no existe: sube CVs con POST /cvs/upload")

        if "--probar" in sys.argv:
            probar_escritura(qd)
    finally:
        qd.close()


def probar_escritura(qd):
    """Crea una coleccion temporal, escribe, filtra por payload y la borra."""
    prueba = COL + "_prueba"
    print(f"\n== Prueba de escritura en '{prueba}' ==")
    try:
        if qd.collection_exists(prueba):
            qd.delete_collection(prueba)
        qd.create_collection(prueba, vectors_config=models.VectorParams(
            size=8, distance=models.Distance.COSINE))
        qd.create_payload_index(prueba, "habilidades", models.PayloadSchemaType.KEYWORD)
        qd.upsert(prueba, points=[
            models.PointStruct(id=1, vector=[1, 0, 0, 0, 0, 0, 0, 0],
                               payload={"candidato": "prueba_a", "habilidades": ["python", "rag"],
                                        "anios": 5, "idiomas": ["ingles"]}),
            models.PointStruct(id=2, vector=[0, 1, 0, 0, 0, 0, 0, 0],
                               payload={"candidato": "prueba_b", "habilidades": ["java"],
                                        "anios": 2, "idiomas": ["espanol"]}),
        ])
        filtro = models.Filter(must=[models.FieldCondition(
            key="habilidades", match=models.MatchAny(any=["python"]))])
        pts = qd.query_points(prueba, query=[1, 0, 0, 0, 0, 0, 0, 0], query_filter=filtro,
                              limit=3, with_payload=True).points
        for p in pts:
            print(f"  encontrado: {p.payload['candidato']} (score {p.score:.3f})")
        ok = len(pts) == 1 and pts[0].payload["candidato"] == "prueba_a"
        print("  resultado:", "OK" if ok else "INESPERADO")
    finally:
        if qd.collection_exists(prueba):
            qd.delete_collection(prueba)
        print("  coleccion de prueba borrada")


if __name__ == "__main__":
    main()
