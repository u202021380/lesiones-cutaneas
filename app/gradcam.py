"""HU04: mapa Grad-CAM (Selvaraju et al., 2017) y superposición sobre la foto."""
import base64
import io

import numpy as np
from PIL import Image


def calcular_gradcam(modelo, x: np.ndarray, indice_clase: int, capa: str) -> np.ndarray:
    """Devuelve un mapa de calor normalizado en [0, 1] del tamaño de la capa (p. ej., 7 x 7)."""
    import tensorflow as tf

    modelo_grad = tf.keras.Model(modelo.inputs, [modelo.get_layer(capa).output, modelo.output])
    with tf.GradientTape() as cinta:
        activaciones, predicciones = modelo_grad([x], training=False)
        puntaje = predicciones[:, indice_clase]
    gradientes = cinta.gradient(puntaje, activaciones)
    pesos = tf.reduce_mean(gradientes, axis=(0, 1, 2))
    mapa = tf.reduce_sum(activaciones[0] * pesos, axis=-1)
    mapa = tf.nn.relu(mapa).numpy()
    maximo = mapa.max()
    return mapa / maximo if maximo > 0 else mapa


def _colormap_jet(v: np.ndarray) -> np.ndarray:
    """Mapa de color tipo jet sin depender de matplotlib (v en [0, 1])."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def superponer(imagen: Image.Image, mapa: np.ndarray, alfa: float = 0.4, ancho_max: int = 640) -> str:
    """Superpone el mapa sobre la foto y devuelve un JPEG en base64."""
    img = imagen.convert("RGB")
    if img.width > ancho_max:
        img = img.resize((ancho_max, round(img.height * ancho_max / img.width)), Image.BILINEAR)
    mapa_img = Image.fromarray(np.uint8(mapa * 255)).resize(img.size, Image.BILINEAR)
    color = Image.fromarray(_colormap_jet(np.asarray(mapa_img, dtype=np.float32) / 255.0))
    mezcla = Image.blend(img, color, alfa)
    buffer = io.BytesIO()
    mezcla.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
