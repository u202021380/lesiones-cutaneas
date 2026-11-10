"""Configuración del sistema.

Los umbrales se leen de variables de entorno para poder ajustarlos sin cambiar
el código. Los valores por defecto son la propuesta inicial del Product Owner
y deben calibrarse con los resultados de HU06 (confianza) y con
scripts/calibrar_calidad.py (nitidez y brillo).
"""
import os

# HU01: reglas de la fotografía
FORMATOS_ADMITIDOS = {"JPEG", "PNG"}
TAMANO_MAXIMO_BYTES = 10 * 1024 * 1024  # 10 MB = 10 485 760 bytes
LADO_MINIMO_PX = 224

# HU02: entrada del modelo
TAMANO_ENTRADA = 224

# HU09: control de calidad (se calculan sobre la imagen en gris de 512 px de ancho)
UMBRAL_NITIDEZ = float(os.getenv("UMBRAL_NITIDEZ", "60"))       # varianza del laplaciano
UMBRAL_BRILLO_MIN = float(os.getenv("UMBRAL_BRILLO_MIN", "40"))  # brillo medio (0-255)
UMBRAL_BRILLO_MAX = float(os.getenv("UMBRAL_BRILLO_MAX", "235"))

# HU12: advertencia de baja confianza
UMBRAL_CONFIANZA = float(os.getenv("UMBRAL_CONFIANZA", "0.60"))

# Modelo
RUTA_MODELO = os.getenv("RUTA_MODELO", "models/modelo.keras")
RUTA_INFO_MODELO = os.getenv("RUTA_INFO_MODELO", "models/modelo_info.json")
USAR_MODELO_SIMULADO = os.getenv("USAR_MODELO_SIMULADO", "0") == "1"

CLASES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
NOMBRES_CLASES = {
    "ACK": "Queratosis actínica",
    "BCC": "Carcinoma basocelular",
    "MEL": "Melanoma",
    "NEV": "Nevo melanocítico",
    "SCC": "Carcinoma epidermoide",
    "SEK": "Queratosis seborreica",
}
CLASES_MALIGNAS = {"BCC", "MEL", "SCC"}
