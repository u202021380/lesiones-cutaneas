"""Casos de prueba del Sprint 1 (Tabla 39): CP01 a CP06, CP08 y CP10 (propuesto)."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from imagenes import MB, ajustar_peso, codificar, textura


def enviar(cliente, datos: bytes, nombre="foto.jpg", tipo="image/jpeg"):
    return cliente.post("/api/v1/evaluaciones", files={"imagen": (nombre, datos, tipo)})


# CP01 - HU01: carga válida (JPG de 2.4 MB y 3024 x 4032 px)
def test_cp01_carga_valida(cliente):
    datos = ajustar_peso(codificar(textura(3024, 4032), "JPG"), int(2.4 * MB))
    r = enviar(cliente, datos)
    assert r.status_code == 200, r.text


# CP02 - HU01: formato no admitido
@pytest.mark.parametrize("formato", ["GIF", "WEBP", "HEIC"])
def test_cp02_formato_no_admitido(cliente, formato):
    r = enviar(cliente, codificar(textura(1080, 1440), formato), f"foto.{formato.lower()}", "image/" + formato.lower())
    assert r.status_code == 422
    assert r.json()["detail"] == "Formato no admitido: use JPG o PNG"


# CP03 - HU01: archivo demasiado grande (JPG de 12 MB) y límite exacto de 10 MB
def test_cp03_archivo_demasiado_grande(cliente):
    datos = ajustar_peso(codificar(textura(1500, 2000), "JPG"), 12 * MB)
    r = enviar(cliente, datos)
    assert r.status_code == 422
    assert "10 MB" in r.json()["detail"]


def test_cp03_limite_10_mb_se_acepta(cliente):
    datos = ajustar_peso(codificar(textura(1500, 2000), "JPG"), 10 * MB)
    assert len(datos) == 10 * MB
    assert enviar(cliente, datos).status_code == 200


# CP04 - HU01: resolución insuficiente (PNG de 200 x 150) y límite de 224 px
def test_cp04_resolucion_insuficiente(cliente):
    r = enviar(cliente, codificar(textura(200, 150), "PNG"), "foto.png", "image/png")
    assert r.status_code == 422
    assert "224 px" in r.json()["detail"]


def test_cp04_limite_224_px_se_acepta(cliente):
    r = enviar(cliente, codificar(textura(224, 300), "PNG"), "foto.png", "image/png")
    assert r.status_code == 200, r.text


# CP05 - HU02: preprocesamiento con valores conocidos
def test_cp05_preprocesamiento():
    from app.preprocesamiento import preprocesar

    arr = np.zeros((300, 400, 3), dtype=np.uint8)
    arr[..., 1] = 255
    arr[..., 2] = 51  # 51 / 127.5 - 1 = -0.6
    x = preprocesar(Image.fromarray(arr))
    assert x.shape == (1, 224, 224, 3)
    assert x.dtype == np.float32
    assert np.allclose(x[..., 0], -1.0)
    assert np.allclose(x[..., 1], 1.0)
    assert np.allclose(x[..., 2], -0.6, atol=1e-6)
    assert x.min() >= -1.0 and x.max() <= 1.0


# CP06 - HU06: partición por paciente sin fuga de datos
def test_cp06_particion_sin_fuga():
    from training.datos import particionar_por_paciente, verificar_sin_fuga

    rng = np.random.default_rng(0)
    filas = []
    clases = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
    for paciente in range(400):
        clase = clases[paciente % 6]
        for _ in range(rng.integers(1, 4)):  # varias fotos por paciente
            filas.append({"patient_id": f"PAT_{paciente}", "diagnostic": clase, "etiqueta": clases.index(clase)})
    df = particionar_por_paciente(pd.DataFrame(filas))
    resultado = verificar_sin_fuga(df)
    assert resultado["sin_fuga"], resultado
    proporcion_test = (df["conjunto"] == "test").mean()
    assert 0.10 < proporcion_test < 0.20


# CP08 - HU03: clasificación de punta a punta
def test_cp08_clasificacion_punta_a_punta(cliente):
    r = enviar(cliente, codificar(textura(1536, 2048, semilla=3), "JPG"))
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["categoria"] in {"ACK", "BCC", "MEL", "NEV", "SCC", "SEK"}
    assert 0.0 <= j["confianza"] <= 1.0
    assert set(j["probabilidades"]) == {"ACK", "BCC", "MEL", "NEV", "SCC", "SEK"}
    assert abs(sum(j["probabilidades"].values()) - 1.0) < 0.001
    assert j["version_modelo"]
    assert isinstance(j["tiempo_procesamiento_ms"], int)
    assert j["confianza"] == max(j["probabilidades"].values())


# CP10 (propuesto) - HU01: imagen dañada
def test_cp10_imagen_truncada(cliente):
    datos = codificar(textura(1536, 2048), "JPG")
    r = enviar(cliente, datos[: len(datos) // 2])
    assert r.status_code == 422
    assert r.json()["detail"] == "La imagen está dañada o no se puede leer"


def test_cp10_archivo_vacio(cliente):
    r = enviar(cliente, b"", "foto.png", "image/png")
    assert r.status_code == 422
    assert r.json()["detail"] == "La imagen está dañada o no se puede leer"


def test_salud(cliente):
    r = cliente.get("/api/v1/salud")
    assert r.status_code == 200
    assert r.json()["modelo_listo"] is True
