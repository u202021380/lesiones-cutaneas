"""Generación de imágenes de prueba con características controladas.

Las imágenes son sintéticas (bloques de colores con bordes nítidos): sirven para
probar las reglas del sistema, no el desempeño del modelo.
"""
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MB = 1024 * 1024


def textura(ancho: int, alto: int, semilla: int = 0, media: float | None = None) -> Image.Image:
    """Bloques de 16 px de colores aleatorios repetidos: nítida, con brillo medio y que comprime bien."""
    rng = np.random.default_rng(semilla)
    bloques = rng.integers(40, 216, size=(16, 16, 3)).astype(np.uint8)
    if media is not None:
        bloques = np.clip(bloques.astype(np.float32) - bloques.mean() + media, 0, 255).astype(np.uint8)
    parche = np.kron(bloques, np.ones((16, 16, 1), dtype=np.uint8))  # 256 x 256
    reps = (alto // 256 + 1, ancho // 256 + 1, 1)
    arr = np.tile(parche, reps)[:alto, :ancho]
    return Image.fromarray(arr)


def codificar(img: Image.Image, formato: str, calidad: int = 90) -> bytes:
    buf = io.BytesIO()
    if formato.upper() in ("JPG", "JPEG"):
        img.convert("RGB").save(buf, format="JPEG", quality=calidad)
    elif formato.upper() == "PNG":
        img.save(buf, format="PNG", compress_level=6)
    elif formato.upper() == "GIF":
        img.convert("P").save(buf, format="GIF")
    elif formato.upper() == "WEBP":
        img.save(buf, format="WEBP", quality=80)
    elif formato.upper() == "HEIC":
        # PIL no escribe HEIC: se simula la cabecera de un archivo HEIC real
        return b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 2048
    else:
        raise ValueError(formato)
    return buf.getvalue()


def ajustar_peso(datos: bytes, peso_objetivo: int) -> bytes:
    """Completa el archivo hasta el peso indicado (los bytes extra van después del fin de la imagen)."""
    if len(datos) < peso_objetivo:
        datos = datos + b"\x00" * (peso_objetivo - len(datos))
    return datos


def borrosa(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=max(6, img.width // 150)))


def oscura(img: Image.Image) -> Image.Image:
    return Image.fromarray((np.asarray(img, dtype=np.float32) * 0.15).astype(np.uint8))


def documento(ancho: int, alto: int) -> Image.Image:
    """Foto de una receta: fondo claro con líneas de texto (no es una lesión)."""
    img = Image.new("RGB", (ancho, alto), (236, 236, 230))
    d = ImageDraw.Draw(img)
    paso = max(12, alto // 40)
    for y in range(paso * 2, alto - paso, paso):
        d.line([(ancho // 10, y), (ancho * 9 // 10, y)], fill=(40, 40, 40), width=max(2, paso // 4))
    return img
