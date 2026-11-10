"""HU09: calibra los umbrales de nitidez y brillo con fotografías reales.

Toma fotos válidas (por ejemplo, el conjunto de prueba de PAD-UFES-20), genera
versiones degradadas (borrosas, oscuras y sobreexpuestas) y mide:
  - el porcentaje de fotos degradadas que se rechazan (meta de HU09: >= 90 %);
  - el porcentaje de fotos originales que se rechazan por error.
También propone un umbral de nitidez a partir de los datos.

Uso:
  python scripts/calibrar_calidad.py --carpeta ruta/a/fotos --salida resultados/calidad.json [--max 300]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import calidad, config  # noqa: E402


def degradar(img: Image.Image, tipo: str, rng) -> Image.Image:
    if tipo == "borrosa":
        return img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(3, 8)) * img.width / 1000))
    arr = np.asarray(img, dtype=np.float32)
    if tipo == "oscura":
        return Image.fromarray(np.clip(arr * rng.uniform(0.08, 0.2), 0, 255).astype(np.uint8))
    if tipo == "sobreexpuesta":
        return Image.fromarray(np.clip(arr * 1.0 + rng.uniform(150, 200), 0, 255).astype(np.uint8))
    raise ValueError(tipo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", required=True)
    ap.add_argument("--salida", default="resultados/calidad.json")
    ap.add_argument("--max", type=int, default=300)
    args = ap.parse_args()

    rutas = sorted(glob.glob(os.path.join(args.carpeta, "**", "*.png"), recursive=True)
                   + glob.glob(os.path.join(args.carpeta, "**", "*.jpg"), recursive=True))[: args.max]
    if not rutas:
        sys.exit("No se encontraron imágenes")
    rng = np.random.default_rng(42)
    filas = []
    for r in rutas:
        with Image.open(r) as im:
            img = im.convert("RGB")
        for tipo in ("original", "borrosa", "oscura", "sobreexpuesta"):
            x = img if tipo == "original" else degradar(img, tipo, rng)
            res = calidad.evaluar_calidad(x)
            filas.append({"tipo": tipo, "nitidez": res.nitidez, "brillo": res.brillo, "aceptada": res.aceptada})

    def tasa(tipo, campo="aceptada", valor=False):
        sub = [f for f in filas if f["tipo"] == tipo]
        return round(100 * sum(f[campo] == valor for f in sub) / len(sub), 1)

    nit_orig = np.array([f["nitidez"] for f in filas if f["tipo"] == "original"])
    nit_borr = np.array([f["nitidez"] for f in filas if f["tipo"] == "borrosa"])
    # Umbral propuesto: rechaza el 90 % de las borrosas
    propuesto = float(np.percentile(nit_borr, 90))

    resumen = {
        "n_fotos": len(rutas),
        "umbrales_usados": {"nitidez": config.UMBRAL_NITIDEZ, "brillo_min": config.UMBRAL_BRILLO_MIN,
                            "brillo_max": config.UMBRAL_BRILLO_MAX},
        "rechazo_de_degradadas_pct": {t: tasa(t) for t in ("borrosa", "oscura", "sobreexpuesta")},
        "rechazo_de_degradadas_total_pct": round(np.mean([tasa(t) for t in ("borrosa", "oscura", "sobreexpuesta")]), 1),
        "rechazo_por_error_de_originales_pct": tasa("original"),
        "nitidez_originales_percentiles_5_50_95": [round(float(np.percentile(nit_orig, q)), 1) for q in (5, 50, 95)],
        "nitidez_borrosas_percentiles_5_50_95": [round(float(np.percentile(nit_borr, q)), 1) for q in (5, 50, 95)],
        "umbral_nitidez_propuesto": round(propuesto, 1),
        "originales_rechazadas_con_umbral_propuesto_pct": round(float((nit_orig < propuesto).mean() * 100), 1),
    }
    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
