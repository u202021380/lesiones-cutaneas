"""Métricas del modelo (HU06, IE4, CP07) y comparación de arquitecturas (HU16)."""
import os
import time

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)

from app.config import CLASES, CLASES_MALIGNAS

IDX_MALIGNAS = [CLASES.index(c) for c in CLASES_MALIGNAS]


def _auc_macro(y, p):
    return roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(len(CLASES))))


def intervalo_bootstrap(y, p, funcion, n=1000, semilla=42):
    rng = np.random.default_rng(semilla)
    valores = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < len(np.unique(y)):
            continue
        valores.append(funcion(y[idx], p[idx]))
    return [round(float(np.percentile(valores, 2.5)), 4), round(float(np.percentile(valores, 97.5)), 4)]


def metricas(y, probs) -> dict:
    y = np.asarray(y)
    pred = probs.argmax(axis=1)
    mc = confusion_matrix(y, pred, labels=list(range(len(CLASES))))
    sensibilidad = recall_score(y, pred, labels=list(range(len(CLASES))), average=None, zero_division=0)
    especificidad = []
    for i in range(len(CLASES)):
        vn = mc.sum() - mc[i, :].sum() - mc[:, i].sum() + mc[i, i]
        fp = mc[:, i].sum() - mc[i, i]
        especificidad.append(vn / (vn + fp) if (vn + fp) else 0.0)

    # Problema binario maligno frente a no maligno (para comparar con DDI, Q5)
    y_bin = np.isin(y, IDX_MALIGNAS).astype(int)
    p_bin = probs[:, IDX_MALIGNAS].sum(axis=1)

    return {
        "n_imagenes": int(len(y)),
        "exactitud": round(float(accuracy_score(y, pred)), 4),
        "exactitud_balanceada": round(float(balanced_accuracy_score(y, pred)), 4),
        "f1_macro": round(float(f1_score(y, pred, average="macro")), 4),
        "auc_roc_macro": round(float(_auc_macro(y, probs)), 4),
        "auc_roc_macro_ic95": intervalo_bootstrap(y, probs, _auc_macro),
        "sensibilidad_por_clase": {c: round(float(s), 4) for c, s in zip(CLASES, sensibilidad)},
        "especificidad_por_clase": {c: round(float(e), 4) for c, e in zip(CLASES, especificidad)},
        "matriz_confusion": {"clases": CLASES, "valores": mc.tolist()},
        "auc_roc_binario_maligno": round(float(roc_auc_score(y_bin, p_bin)), 4),
        "cumple_ie4": bool(accuracy_score(y, pred) >= 0.85 and _auc_macro(y, probs) >= 0.85),
    }


def analisis_confianza(y, probs, umbrales=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> list:
    """Para elegir el umbral de HU12: cobertura y exactitud por encima de cada umbral."""
    y = np.asarray(y)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    filas = []
    for u in umbrales:
        sobre = conf >= u
        filas.append({
            "umbral": u,
            "porcentaje_con_advertencia": round(float((~sobre).mean() * 100), 1),
            "exactitud_sin_advertencia": round(float((pred[sobre] == y[sobre]).mean()), 4) if sobre.any() else None,
            "exactitud_con_advertencia": (round(float((pred[~sobre] == y[~sobre]).mean()), 4)
                                          if (~sobre).any() else None),
        })
    return filas


def tiempo_inferencia_cpu(ruta_modelo: str, repeticiones: int = 50) -> dict:
    """Mediana del tiempo de inferencia de una imagen en CPU (ms) y tamaño del archivo (MB)."""
    import keras
    import tensorflow as tf

    with tf.device("/CPU:0"):
        modelo = keras.models.load_model(ruta_modelo, compile=False)
        x = np.random.uniform(-1, 1, (1, 224, 224, 3)).astype(np.float32)
        for _ in range(5):
            modelo(x, training=False)
        tiempos = []
        for _ in range(repeticiones):
            t = time.perf_counter()
            modelo(x, training=False)
            tiempos.append((time.perf_counter() - t) * 1000)
    return {"mediana_ms_cpu": round(float(np.median(tiempos)), 1),
            "tamano_mb": round(os.path.getsize(ruta_modelo) / 1024 / 1024, 1)}


def decidir_arquitectura(m_v2: dict, m_v3: dict) -> str:
    """Regla de la decisión D1: MobileNetV2 se mantiene salvo que no cumpla IE4 y V3-Large sí."""
    if m_v2["cumple_ie4"]:
        return "MobileNetV2 (cumple IE4; se mantiene la decisión D1)"
    if m_v3["cumple_ie4"]:
        return "MobileNetV3Large (se activa la contingencia del riesgo R2: V2 no cumple IE4 y V3-Large sí)"
    return "Ninguna cumple IE4: se mantiene MobileNetV2 y se aplica la mitigación del riesgo R2 con el PO"
