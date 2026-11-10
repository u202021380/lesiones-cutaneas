"""HU09: control de calidad de la fotografía (nitidez y brillo).

La nitidez se mide con la varianza del laplaciano y el brillo con la media de
la imagen en escala de grises. Ambas se calculan sobre una copia de 512 px de
ancho para que no dependan de la resolución de la cámara.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app import config

ANCHO_ANALISIS = 512

MSG_BORROSA = "La fotografía está borrosa: enfoque la lesión y vuelva a tomarla"
MSG_OSCURA = "La fotografía está muy oscura: mejore la iluminación y vuelva a tomarla"
MSG_SOBREEXPUESTA = "La fotografía tiene demasiada luz: evite el flash directo y vuelva a tomarla"


@dataclass
class ResultadoCalidad:
    nitidez: float
    brillo: float
    aceptada: bool
    motivo: str | None = None
    codigo: str | None = None


def _gris_reducida(imagen: Image.Image) -> np.ndarray:
    ancho, alto = imagen.size
    gris = imagen.convert("L")
    if ancho > ANCHO_ANALISIS:  # solo se reduce; ampliar suavizaría la imagen
        gris = gris.resize((ANCHO_ANALISIS, max(1, round(alto * ANCHO_ANALISIS / ancho))), Image.BILINEAR)
    return np.asarray(gris, dtype=np.float32)


def varianza_laplaciano(g: np.ndarray) -> float:
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1])
    return float(lap.var())


def evaluar_calidad(imagen: Image.Image) -> ResultadoCalidad:
    g = _gris_reducida(imagen)
    nitidez = varianza_laplaciano(g)
    brillo = float(g.mean())
    if brillo < config.UMBRAL_BRILLO_MIN:
        return ResultadoCalidad(nitidez, brillo, False, MSG_OSCURA, "oscura")
    if brillo > config.UMBRAL_BRILLO_MAX:
        return ResultadoCalidad(nitidez, brillo, False, MSG_SOBREEXPUESTA, "sobreexpuesta")
    if nitidez < config.UMBRAL_NITIDEZ:
        return ResultadoCalidad(nitidez, brillo, False, MSG_BORROSA, "borrosa")
    return ResultadoCalidad(nitidez, brillo, True)
