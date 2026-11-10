"""HU01: validación de la fotografía cargada.

Reglas del contrato (Tabla 38): JPG o PNG, hasta 10 MB, lado menor de al menos
224 px e imagen legible. Si no cumple, se responde 422 con el motivo.
"""
import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from app import config

MSG_FORMATO = "Formato no admitido: use JPG o PNG"
MSG_TAMANO = "El archivo supera el tamaño máximo de 10 MB"
MSG_RESOLUCION = "Resolución insuficiente: el lado menor debe medir al menos 224 px"
MSG_DANADA = "La imagen está dañada o no se puede leer"


class FotoInvalida(Exception):
    """Error de validación con un mensaje para el profesional de salud."""

    def __init__(self, motivo: str, codigo: str):
        super().__init__(motivo)
        self.motivo = motivo
        self.codigo = codigo


@dataclass
class FotoValidada:
    imagen: Image.Image  # RGB, con la orientación EXIF aplicada
    formato: str
    ancho: int
    alto: int
    peso_bytes: int


def validar_foto(contenido: bytes) -> FotoValidada:
    peso = len(contenido)
    if peso == 0:
        raise FotoInvalida(MSG_DANADA, "imagen_danada")
    if peso > config.TAMANO_MAXIMO_BYTES:
        raise FotoInvalida(MSG_TAMANO, "tamano")

    try:
        with Image.open(io.BytesIO(contenido)) as img:
            formato = img.format
    except UnidentifiedImageError:
        raise FotoInvalida(MSG_FORMATO, "formato")
    except Exception:
        raise FotoInvalida(MSG_DANADA, "imagen_danada")

    if formato not in config.FORMATOS_ADMITIDOS:
        raise FotoInvalida(MSG_FORMATO, "formato")

    try:
        img = Image.open(io.BytesIO(contenido))
        img.load()  # decodifica la imagen completa: detecta archivos truncados
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except Exception:
        raise FotoInvalida(MSG_DANADA, "imagen_danada")

    ancho, alto = img.size
    if min(ancho, alto) < config.LADO_MINIMO_PX:
        raise FotoInvalida(MSG_RESOLUCION, "resolucion")

    return FotoValidada(imagen=img, formato=formato, ancho=ancho, alto=alto, peso_bytes=peso)
