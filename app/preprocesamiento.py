"""HU02: preprocesamiento de la fotografía.

Esta misma función se usa en el entrenamiento (training/datos.py) y en la API,
para que el modelo reciba la misma entrada en ambos casos.
"""
import numpy as np
from PIL import Image

from app import config


def redimensionar(imagen: Image.Image, tamano: int = config.TAMANO_ENTRADA) -> np.ndarray:
    """Convierte a RGB y redimensiona a 224 x 224 (uint8). Se usa al entrenar y en la API."""
    img = imagen.convert("RGB").resize((tamano, tamano), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def normalizar(x):
    """Lleva los valores de [0, 255] a [-1, 1], como mobilenet_v2.preprocess_input."""
    return x / 127.5 - 1.0


def preprocesar(imagen: Image.Image, tamano: int = config.TAMANO_ENTRADA) -> np.ndarray:
    """Devuelve un tensor de 1 x 224 x 224 x 3 con valores en [-1, 1]."""
    x = normalizar(redimensionar(imagen, tamano).astype(np.float32))
    return x[np.newaxis, ...]
