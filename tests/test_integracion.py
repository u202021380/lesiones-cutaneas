"""Pruebas con el modelo real entrenado (CP07 y CP09). Se omiten si no hay modelo.

Ejecutar con:  USAR_MODELO_SIMULADO=0 pytest tests/test_integracion.py -v
"""
import json
import os
import statistics
import time
from pathlib import Path

import pytest

from imagenes import ajustar_peso, codificar, textura

RAIZ = Path(__file__).parent.parent
METRICAS = RAIZ / "resultados" / "metricas_MobileNetV2.json"
MODELO = RAIZ / "models" / "modelo.keras"
DATOS = json.loads((Path(__file__).parent / "data" / "atenciones_sinteticas_sprint1.json").read_text(encoding="utf-8"))


@pytest.mark.skipif(not METRICAS.exists(),
                    reason="Falta resultados/metricas_MobileNetV2.json (se genera en el notebook)")
def test_cp07_desempeno_en_prueba_interna():
    m = json.loads(METRICAS.read_text(encoding="utf-8"))["prueba"]
    assert m["exactitud"] >= 0.85, f"Exactitud {m['exactitud']}"
    assert m["auc_roc_macro"] >= 0.85, f"AUC-ROC {m['auc_roc_macro']}"


@pytest.mark.skipif(os.getenv("USAR_MODELO_SIMULADO", "1") == "1" or not MODELO.exists(),
                    reason="CP09 requiere el modelo real: USAR_MODELO_SIMULADO=0 y models/modelo.keras")
def test_cp09_tiempo_de_respuesta(cliente):
    lote = [f for a in DATOS["atenciones"] for f in a["fotos"] if "CP09" in f["casos_prueba"]]
    assert len(lote) == 30
    tiempos = []
    for i, f in enumerate(lote):
        datos = ajustar_peso(codificar(textura(f["ancho_px"], f["alto_px"], i), f["formato"]), f["peso_bytes"])
        mime = "image/jpeg" if f["formato"] == "JPG" else "image/png"
        t = time.perf_counter()
        r = cliente.post("/api/v1/evaluaciones", files={"imagen": (f["archivo"], datos, mime)})
        tiempos.append(time.perf_counter() - t)
        assert r.status_code == 200
    mediana = statistics.median(tiempos)
    print(f"\nCP09: mediana = {mediana:.2f} s (n = 30)")
    assert mediana <= 5.0
