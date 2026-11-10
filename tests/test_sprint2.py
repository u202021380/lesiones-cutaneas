"""Pruebas del Sprint 2: HU04 (Grad-CAM), HU09 (calidad), HU11 (guía) y HU12 (baja confianza)."""
import base64
import io

import numpy as np
import pytest
from PIL import Image

from imagenes import borrosa, codificar, oscura, textura


def enviar(cliente, datos: bytes, nombre="foto.jpg", tipo="image/jpeg"):
    return cliente.post("/api/v1/evaluaciones", files={"imagen": (nombre, datos, tipo)})


# HU09: control de calidad
def test_hu09_rechaza_foto_borrosa(cliente):
    r = enviar(cliente, codificar(borrosa(textura(1536, 2048)), "JPG"))
    assert r.status_code == 422
    assert r.json()["codigo"] == "borrosa"


def test_hu09_rechaza_foto_oscura(cliente):
    r = enviar(cliente, codificar(oscura(textura(1536, 2048)), "JPG"))
    assert r.status_code == 422
    assert r.json()["codigo"] == "oscura"


def test_hu09_acepta_foto_nitida(cliente):
    assert enviar(cliente, codificar(textura(1536, 2048), "JPG")).status_code == 200


def test_hu09_metricas_independientes_de_la_resolucion():
    from app.calidad import evaluar_calidad

    chica = evaluar_calidad(textura(1024, 768))
    grande = evaluar_calidad(textura(4096, 3072))
    assert chica.aceptada and grande.aceptada
    assert abs(chica.brillo - grande.brillo) < 5


# HU04: mapa Grad-CAM en el 100 % de las predicciones
def test_hu04_respuesta_incluye_mapa(cliente):
    j = enviar(cliente, codificar(textura(1536, 2048, semilla=5), "JPG")).json()
    assert j["mapa_gradcam"].startswith("data:image/jpeg;base64,")
    jpg = base64.b64decode(j["mapa_gradcam"].split(",", 1)[1])
    img = Image.open(io.BytesIO(jpg))
    assert img.format == "JPEG" and img.width <= 640
    assert len(jpg) < 300_000


def test_hu04_gradcam_con_mobilenetv2_real():
    pytest.importorskip("tensorflow")
    from app.gradcam import calcular_gradcam, superponer
    from training.modelos import construir_modelo

    modelo, _, capa = construir_modelo("MobileNetV2", pesos=None)
    x = np.random.default_rng(0).uniform(-1, 1, (1, 224, 224, 3)).astype(np.float32)
    mapa = calcular_gradcam(modelo, x, 0, capa)
    assert mapa.shape == (7, 7)
    assert mapa.min() >= 0.0 and mapa.max() <= 1.0
    assert superponer(textura(800, 600), mapa)


# HU12: advertencia de baja confianza
def test_hu12_advertencia_con_confianza_baja(cliente):
    # Con el modelo simulado, una imagen de colores equilibrados da probabilidades parejas
    j = enviar(cliente, codificar(textura(1536, 2048, media=127.5), "JPG")).json()
    assert j["confianza"] < 0.60
    assert j["advertencia_baja_confianza"] is True
    assert j["mensaje_advertencia"]


def test_hu12_sin_advertencia_con_confianza_alta(cliente):
    arr = np.asarray(textura(1536, 2048), dtype=np.float32)
    arr[..., 0] = np.clip(arr[..., 0] + 80, 0, 255)   # más rojo y menos verde
    arr[..., 1] = np.clip(arr[..., 1] - 80, 0, 255)
    j = enviar(cliente, codificar(Image.fromarray(arr.astype(np.uint8)), "JPG")).json()
    assert j["confianza"] >= 0.60
    assert j["advertencia_baja_confianza"] is False


# HU11: guía para tomar la foto en la interfaz
def test_hu11_guia_en_la_interfaz(cliente):
    html = cliente.get("/").text
    for texto in ("Distancia", "Iluminación", "Enfoque", "Una lesión por foto", "Ocultar guía"):
        assert texto in html
