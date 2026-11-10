"""MC2 / CP09: mide el tiempo de respuesta del sistema desplegado.

Envía N fotografías al endpoint y reporta la mediana del tiempo total (lo que
espera el profesional) y del tiempo de procesamiento en el servidor.

Uso:
  python scripts/medir_rendimiento.py --url https://<tu-espacio>.hf.space --carpeta fotos/ --n 30
"""
import argparse
import glob
import json
import os
import statistics
import time

import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--carpeta", required=True, help="Fotos JPG o PNG de 2 a 5 MB")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--salida", default="resultados/rendimiento.json")
    args = ap.parse_args()

    rutas = sorted(glob.glob(os.path.join(args.carpeta, "*.jpg")) + glob.glob(os.path.join(args.carpeta, "*.png")))
    rutas = (rutas * (args.n // max(1, len(rutas)) + 1))[: args.n]
    url = args.url.rstrip("/") + "/api/v1/evaluaciones"
    total, servidor, errores = [], [], 0
    with httpx.Client(timeout=60) as cliente:
        for r in rutas:
            with open(r, "rb") as f:
                datos = f.read()
            mime = "image/png" if r.lower().endswith(".png") else "image/jpeg"
            t = time.perf_counter()
            resp = cliente.post(url, files={"imagen": (os.path.basename(r), datos, mime)})
            total.append(time.perf_counter() - t)
            if resp.status_code == 200:
                servidor.append(resp.json()["tiempo_procesamiento_ms"] / 1000)
            else:
                errores += 1
    resumen = {
        "url": args.url, "n": len(total), "errores": errores,
        "mediana_total_s": round(statistics.median(total), 2),
        "p90_total_s": round(sorted(total)[int(0.9 * (len(total) - 1))], 2),
        "mediana_servidor_s": round(statistics.median(servidor), 2) if servidor else None,
        "cumple_mc2": statistics.median(total) <= 5.0,
    }
    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2)
    print(json.dumps(resumen, indent=2))


if __name__ == "__main__":
    main()
