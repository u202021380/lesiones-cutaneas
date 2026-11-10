"""API del sistema de clasificación preliminar de lesiones cutáneas.

POST /api/v1/evaluaciones  -> HU01, HU02, HU03, HU04, HU09, HU12
GET  /api/v1/salud         -> estado del servicio y del modelo
GET  /                     -> interfaz web (HU11: guía para tomar la foto)
"""
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.calidad import evaluar_calidad
from app.gradcam import superponer
from app.modelo import obtener_modelo
from app.preprocesamiento import preprocesar
from app.validacion import FotoInvalida, MSG_TAMANO, validar_foto

VERSION_API = "0.2.0"  # 0.1 = Sprint 1, 0.2 = Sprint 2
AVISO = ("Resultado preliminar de apoyo a la evaluación inicial. No es un diagnóstico: "
         "la decisión clínica corresponde al profesional de salud.")

log = logging.getLogger("evaluaciones")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Clasificación preliminar de lesiones cutáneas", version=VERSION_API)


def _error_422(motivo: str, codigo: str) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": motivo, "codigo": codigo})


@app.get("/api/v1/salud")
def salud():
    modelo = obtener_modelo()
    return {
        "estado": "ok",
        "version_api": VERSION_API,
        "modelo_listo": modelo is not None,
        "version_modelo": getattr(modelo, "version", None),
        "arquitectura": getattr(modelo, "arquitectura", None),
        "modelo_simulado": getattr(modelo, "simulado", None),
        "umbrales": {
            "confianza": config.UMBRAL_CONFIANZA,
            "nitidez": config.UMBRAL_NITIDEZ,
            "brillo_min": config.UMBRAL_BRILLO_MIN,
            "brillo_max": config.UMBRAL_BRILLO_MAX,
        },
    }


@app.post("/api/v1/evaluaciones")
async def crear_evaluacion(imagen: UploadFile = File(...)):
    inicio = time.perf_counter()
    id_evaluacion = str(uuid.uuid4())

    contenido = await imagen.read(config.TAMANO_MAXIMO_BYTES + 1)
    if len(contenido) > config.TAMANO_MAXIMO_BYTES:
        log.info("evaluacion=%s rechazada=tamano", id_evaluacion)
        return _error_422(MSG_TAMANO, "tamano")

    # HU01: validación
    try:
        foto = validar_foto(contenido)
    except FotoInvalida as e:
        log.info("evaluacion=%s rechazada=%s", id_evaluacion, e.codigo)
        return _error_422(e.motivo, e.codigo)

    # HU09: control de calidad
    calidad = evaluar_calidad(foto.imagen)
    if not calidad.aceptada:
        log.info("evaluacion=%s rechazada=%s", id_evaluacion, calidad.codigo)
        return _error_422(calidad.motivo, calidad.codigo)

    modelo = obtener_modelo()
    if modelo is None:
        return JSONResponse(status_code=503, content={"detail": "El modelo no está disponible"})

    # HU02 y HU03: preprocesamiento y clasificación
    x = preprocesar(foto.imagen)
    probs = modelo.predecir(x)
    indice = int(probs.argmax())
    categoria = modelo.clases[indice]
    confianza = float(probs[indice])

    # HU04: Grad-CAM
    mapa = modelo.mapa_gradcam(x, indice)
    mapa_jpg = superponer(foto.imagen, mapa)

    # HU12: advertencia de baja confianza
    baja_confianza = confianza < config.UMBRAL_CONFIANZA
    tiempo_ms = round((time.perf_counter() - inicio) * 1000)
    log.info("evaluacion=%s categoria=%s confianza=%.3f tiempo_ms=%d",
             id_evaluacion, categoria, confianza, tiempo_ms)

    return {
        "id_evaluacion": id_evaluacion,
        "categoria": categoria,
        "nombre_categoria": config.NOMBRES_CLASES.get(categoria, categoria),
        "grupo": "Cáncer" if categoria in config.CLASES_MALIGNAS else "No cáncer",
        "confianza": round(confianza, 4),
        "probabilidades": {c: round(float(p), 4) for c, p in zip(modelo.clases, probs)},
        "advertencia_baja_confianza": baja_confianza,
        "mensaje_advertencia": (
            "Confianza baja: el resultado es incierto. Repita la foto o apóyese solo en su evaluación clínica."
            if baja_confianza else None),
        "calidad": {"nitidez": round(calidad.nitidez, 1), "brillo": round(calidad.brillo, 1)},
        "mapa_gradcam": "data:image/jpeg;base64," + mapa_jpg,
        "version_modelo": modelo.version,
        "modelo_simulado": modelo.simulado,
        "tiempo_procesamiento_ms": tiempo_ms,
        "aviso": AVISO,
    }


_static = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=_static, html=True), name="web")
