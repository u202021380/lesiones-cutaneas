"""Entrenamiento en dos fases con Transfer Learning (HU06 y HU16)."""
import json
import os
import time

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from training.modelos import construir_modelo, descongelar_ultimas

CAPAS_A_AJUSTAR = {"MobileNetV2": 40, "MobileNetV3Large": 40}


def pesos_por_clase(y) -> dict:
    clases = np.unique(y)
    pesos = compute_class_weight("balanced", classes=clases, y=y)
    return {int(c): float(p) for c, p in zip(clases, pesos)}


def entrenar(arquitectura, ds_train, ds_val, y_train, carpeta_salida,
             epocas_fase1=15, epocas_fase2=30, lr_fase1=1e-3, lr_fase2=1e-5, semilla=42, pesos_base="imagenet"):
    import keras

    keras.utils.set_random_seed(semilla)
    modelo, base, capa_gradcam = construir_modelo(arquitectura, pesos=pesos_base)
    pesos = pesos_por_clase(y_train)
    metricas = ["accuracy", keras.metrics.AUC(name="auc", multi_label=False)]
    historial = {}
    os.makedirs(carpeta_salida, exist_ok=True)
    inicio = time.time()

    # Fase 1: solo la cabeza de clasificación
    modelo.compile(optimizer=keras.optimizers.Adam(lr_fase1), loss="categorical_crossentropy", metrics=metricas)
    h1 = modelo.fit(ds_train, validation_data=ds_val, epochs=epocas_fase1, class_weight=pesos, verbose=2,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4,
                                                             restore_best_weights=True)])
    historial["fase1"] = {k: [float(v) for v in vals] for k, vals in h1.history.items()}

    # Fase 2: ajuste fino de los últimos bloques con tasa de aprendizaje baja
    descongelar_ultimas(base, CAPAS_A_AJUSTAR[arquitectura])
    modelo.compile(optimizer=keras.optimizers.Adam(lr_fase2), loss="categorical_crossentropy", metrics=metricas)
    h2 = modelo.fit(ds_train, validation_data=ds_val, epochs=epocas_fase2, class_weight=pesos, verbose=2,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6,
                                                             restore_best_weights=True)])
    historial["fase2"] = {k: [float(v) for v in vals] for k, vals in h2.history.items()}
    historial["minutos"] = round((time.time() - inicio) / 60, 1)
    historial["pesos_por_clase"] = pesos

    ruta_modelo = os.path.join(carpeta_salida, f"{arquitectura}.keras")
    modelo.save(ruta_modelo)
    with open(os.path.join(carpeta_salida, f"historial_{arquitectura}.json"), "w", encoding="utf-8") as f:
        json.dump(historial, f, indent=2)
    return modelo, capa_gradcam, ruta_modelo, historial
