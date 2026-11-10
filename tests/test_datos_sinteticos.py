"""Ejecuta las 87 solicitudes del conjunto de datos sintéticos contra la API.

Cada foto se genera con el formato, el peso y la resolución del JSON. El
resultado esperado se ajusta al Sprint 2: las fotos borrosas ahora se
rechazan (HU09); las que no muestran una lesión todavía se aceptan.
"""
import json
from pathlib import Path

import pytest

from imagenes import ajustar_peso, borrosa, codificar, documento, textura

RUTA = Path(__file__).parent / "data" / "atenciones_sinteticas_sprint1.json"
DATOS = json.loads(RUTA.read_text(encoding="utf-8"))
FOTOS = [(a, f) for a in DATOS["atenciones"] for f in a["fotos"]]
LIMITE = 10 * 1024 * 1024


def construir(foto: dict, semilla: int) -> bytes:
    fmt, peso = foto["formato"], foto["peso_bytes"]
    tipo = foto["tipo_caso"]
    if tipo.startswith("Inválida: dañada"):
        if peso == 0:
            return b""
        datos = codificar(textura(1536, 2048, semilla), "JPG")
        return datos[: len(datos) // 2]
    ancho, alto = foto["ancho_px"], foto["alto_px"]
    if tipo == "Calidad: borrosa":
        img = borrosa(textura(ancho, alto, semilla))
    elif tipo == "Calidad: no corresponde a una lesión":
        img = documento(ancho, alto)
    else:
        img = textura(ancho, alto, semilla)
    datos = codificar(img, fmt)
    if fmt in ("JPG", "PNG") and peso <= LIMITE and len(datos) > LIMITE:
        datos = codificar(img.resize((ancho, alto)), "JPG", calidad=70)
    return ajustar_peso(datos, peso)


def esperado_sprint2(foto: dict) -> int:
    if foto["tipo_caso"] == "Calidad: borrosa":
        return 422
    return foto["resultado_esperado"]["http"]


@pytest.mark.parametrize("atencion,foto", FOTOS, ids=[f["solicitud_id"] for _, f in FOTOS])
def test_solicitud_sintetica(cliente, atencion, foto):
    datos = construir(foto, semilla=int(foto["solicitud_id"][3:]))
    mime = {"JPG": "image/jpeg", "PNG": "image/png"}.get(foto["formato"], "application/octet-stream")
    r = cliente.post("/api/v1/evaluaciones", files={"imagen": (foto["archivo"], datos, mime)})
    assert r.status_code == esperado_sprint2(foto), (foto["tipo_caso"], r.text[:200])
    if r.status_code == 422 and "detail" in foto["resultado_esperado"]:
        assert r.json()["detail"] == foto["resultado_esperado"]["detail"]
