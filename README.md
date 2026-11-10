# Sistema de clasificación preliminar de lesiones cutáneas

Transfer Learning con **MobileNetV2** y explicabilidad visual con **Grad-CAM**, para la evaluación inicial en la consulta externa de un centro de salud MINSA (categoría I-4).

> Resultado preliminar de apoyo. **No es un diagnóstico**: la decisión clínica corresponde al profesional de salud.

## Qué historia de usuario implementa cada archivo

| Sprint | HU | Qué hace | Archivo |
|---|---|---|---|
| 1 | HU06 | Entrena y valida MobileNetV2 con partición por paciente | `notebooks/entrenamiento_HU06_HU16.ipynb`, `training/` |
| 1 | HU01 | Valida la foto: JPG o PNG, ≤ 10 MB, lado menor ≥ 224 px, legible | `app/validacion.py` |
| 1 | HU02 | Preprocesa la foto (224 × 224, valores en [−1, 1]); la misma función se usa al entrenar | `app/preprocesamiento.py` |
| 1 | HU03 | Clasifica y devuelve la categoría, la confianza y las seis probabilidades | `app/main.py`, `app/modelo.py` |
| 2 | HU04 | Mapa Grad-CAM sobre la foto | `app/gradcam.py` |
| 2 | HU09 | Rechaza fotos borrosas, oscuras o sobreexpuestas | `app/calidad.py`, `scripts/calibrar_calidad.py` |
| 2 | HU11 | Guía de cuatro indicaciones para tomar la foto, que se puede ocultar | `app/static/index.html` |
| 2 | HU12 | Advertencia cuando la confianza es menor que el umbral | `app/main.py` (`UMBRAL_CONFIANZA`) |
| 2 | HU16 | Compara MobileNetV2 con MobileNetV3-Large con el mismo protocolo | notebook, sección 8 |

## API

`POST /api/v1/evaluaciones` (multipart, campo `imagen`)

- **200**: `categoria`, `nombre_categoria`, `grupo`, `confianza`, `probabilidades` (6), `advertencia_baja_confianza`, `mensaje_advertencia`, `calidad`, `mapa_gradcam` (JPEG en base64), `version_modelo`, `modelo_simulado`, `tiempo_procesamiento_ms`, `aviso`.
- **422**: `detail` (motivo para el profesional) y `codigo` (`formato`, `tamano`, `resolucion`, `imagen_danada`, `borrosa`, `oscura`, `sobreexpuesta`).
- **503**: el modelo no está cargado.

`GET /api/v1/salud`: estado del servicio, versión del modelo y umbrales.

## 1. Pruebas automatizadas (no necesitan el modelo)

```bash
pip install -r requirements-dev.txt
pytest --cov=app --cov-report=term-missing --cov-report=html
flake8 app training tests scripts
```

Las pruebas usan un **modelo simulado** (`USAR_MODELO_SIMULADO=1`): sirven para verificar las reglas del sistema, no el desempeño del modelo. Incluyen CP01 a CP06, CP08, CP10 (imagen dañada, propuesto), las pruebas del Sprint 2 y las 87 solicitudes del conjunto de datos sintéticos (`tests/data/atenciones_sinteticas_sprint1.json`). CP07 y CP09 están en `tests/test_integracion.py` y se ejecutan con el modelo real.

## 2. Entrenamiento (Google Colab con GPU)

1. Abre `notebooks/entrenamiento_HU06_HU16.ipynb` en Colab y elige **GPU T4**.
2. En la celda 1, pon la URL de tu repositorio (o sube `lesiones-cutaneas.zip`).
3. Ejecuta todas las celdas. Al final se descarga `resultados_entrenamiento.zip` con:
   - `models/modelo.keras` y `models/modelo_info.json`;
   - `resultados/metricas_MobileNetV2.json`, `metricas_MobileNetV3Large.json` y `comparacion_HU16.json`;
   - `resultados/particion.csv` (evidencia de CP06), curvas, matrices de confusión, ejemplos de Grad-CAM y `calidad_HU09.json`.
4. Descomprime el zip en la raíz del repositorio y haz commit de `resultados/` (y de `models/` con Git LFS o súbelo directo al Space).
5. Con el modelo real:
   ```bash
   USAR_MODELO_SIMULADO=0 pytest tests/test_integracion.py -v -s   # CP07 y CP09
   ```

Cambia en `requirements.txt` la versión de `tensorflow-cpu` por la que figura en `models/modelo_info.json`.

## 3. Ejecutar en local

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 7860
# abre http://localhost:7860
```

Sin modelo entrenado puedes probar la interfaz con `USAR_MODELO_SIMULADO=1` (aparece un aviso de «modo de prueba»).

## 4. Despliegue en Hugging Face Spaces (gratis, CPU)

1. Crea un Space nuevo: **SDK = Docker**, hardware **CPU basic**.
2. Sube los archivos del repositorio (incluido `models/modelo.keras`; la web de Hugging Face lo guarda con LFS automáticamente).
3. Agrega al inicio de `README.md` del Space:
   ```yaml
   ---
   title: Lesiones cutáneas
   sdk: docker
   app_port: 7860
   ---
   ```
4. Espera a que termine la construcción y abre `https://<usuario>-<space>.hf.space`.
5. Mide el tiempo de respuesta (MC2 y CP09):
   ```bash
   python scripts/medir_rendimiento.py --url https://<usuario>-<space>.hf.space --carpeta fotos_prueba/ --n 30
   ```

## 5. Seguridad (MC4)

```bash
docker run -t ghcr.io/zaproxy/zaproxy:stable zap-baseline.py -t https://<usuario>-<space>.hf.space -r zap.html
```

## Privacidad

La API no guarda las fotos ni registra datos del paciente; los registros solo contienen el identificador de la evaluación, la categoría, la confianza y el tiempo (DoD 7).

## Datos

- PAD-UFES-20: Pacheco, A. G. C., et al. (2020). *Data in Brief, 32*, 106221. https://doi.org/10.17632/zr7vgbcyr2.1 (CC BY 4.0).
- Los datos de `tests/data/` son **sintéticos** y no corresponden a pacientes reales.
