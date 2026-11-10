"""Carga del modelo y predicción (HU03), con Grad-CAM (HU04).

ModeloKeras usa el modelo entrenado en el notebook (models/modelo.keras).
ModeloSimulado solo sirve para las pruebas automáticas y para probar la
interfaz sin el modelo: sus resultados NO son clasificaciones reales y la API
lo indica con "modelo_simulado": true.
"""
import json
import os

import numpy as np

from app import config


class ModeloKeras:
    simulado = False

    def __init__(self, ruta_modelo: str, ruta_info: str):
        import keras

        self.modelo = keras.models.load_model(ruta_modelo, compile=False)
        info = {}
        if os.path.exists(ruta_info):
            with open(ruta_info, encoding="utf-8") as f:
                info = json.load(f)
        self.clases = info.get("clases", config.CLASES)
        self.version = info.get("version", os.path.basename(ruta_modelo))
        self.arquitectura = info.get("arquitectura", "MobileNetV2")
        self.capa_gradcam = info.get("capa_gradcam", "out_relu")

    def predecir(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.modelo([x], training=False))[0]

    def mapa_gradcam(self, x: np.ndarray, indice: int) -> np.ndarray:
        from app.gradcam import calcular_gradcam

        return calcular_gradcam(self.modelo, x, indice, self.capa_gradcam)


class ModeloSimulado:
    """Modelo de reemplazo para pruebas: determinista y sin TensorFlow."""

    simulado = True
    clases = config.CLASES
    version = "simulado-0.1"
    arquitectura = "simulado"

    def predecir(self, x: np.ndarray) -> np.ndarray:
        canales = x[0].mean(axis=(0, 1))  # media por canal en [-1, 1]
        logits = np.array([canales[0], canales[1], -canales[2], canales[0] - canales[1],
                           canales[2], -canales[0]], dtype=np.float64) * 3.0
        e = np.exp(logits - logits.max())
        return (e / e.sum()).astype(np.float32)

    def mapa_gradcam(self, x: np.ndarray, indice: int) -> np.ndarray:
        yy, xx = np.mgrid[0:7, 0:7]
        return np.exp(-((yy - 3) ** 2 + (xx - 3) ** 2) / 4.0).astype(np.float32)


_modelo = None


def obtener_modelo():
    global _modelo
    if _modelo is None:
        if config.USAR_MODELO_SIMULADO:
            _modelo = ModeloSimulado()
        elif os.path.exists(config.RUTA_MODELO):
            _modelo = ModeloKeras(config.RUTA_MODELO, config.RUTA_INFO_MODELO)
    return _modelo


def reiniciar_modelo():
    global _modelo
    _modelo = None
