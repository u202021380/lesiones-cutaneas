"""Carga de PAD-UFES-20, partición por paciente (HU06, CP06) y pipelines de datos."""
import glob
import os
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

from app.config import CLASES
from app.preprocesamiento import normalizar, redimensionar


def descomprimir_todo(carpeta: str) -> None:
    """Descomprime en su sitio los .zip (también los anidados) de la carpeta."""
    hecho = set()
    while True:
        pendientes = [z for z in glob.glob(os.path.join(carpeta, "**", "*.zip"), recursive=True) if z not in hecho]
        if not pendientes:
            return
        for z in pendientes:
            with zipfile.ZipFile(z) as f:
                f.extractall(os.path.dirname(z))
            hecho.add(z)


def cargar_metadatos(carpeta: str) -> pd.DataFrame:
    rutas = glob.glob(os.path.join(carpeta, "**", "metadata.csv"), recursive=True)
    if not rutas:
        raise FileNotFoundError("No se encontró metadata.csv de PAD-UFES-20")
    df = pd.read_csv(rutas[0])
    imagenes = {os.path.basename(p): p for p in glob.glob(os.path.join(carpeta, "**", "*.png"), recursive=True)}
    df["ruta"] = df["img_id"].map(imagenes)
    faltan = df["ruta"].isna().sum()
    if faltan:
        print(f"Aviso: {faltan} imágenes de metadata.csv no se encontraron y se descartan")
        df = df.dropna(subset=["ruta"])
    df = df[df["diagnostic"].isin(CLASES)].reset_index(drop=True)
    df["etiqueta"] = df["diagnostic"].map({c: i for i, c in enumerate(CLASES)})
    return df


def particionar_por_paciente(df: pd.DataFrame, semilla: int = 42) -> pd.DataFrame:
    """Asigna train/val/test (70/15/15) sin que un paciente aparezca en dos conjuntos.

    Usa 20 pliegues estratificados por diagnóstico y agrupados por paciente:
    14 pliegues para entrenamiento, 3 para validación y 3 para prueba.
    """
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=semilla)
    pliegue = np.zeros(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df["etiqueta"], groups=df["patient_id"])):
        pliegue[idx] = k
    conjunto = np.where(pliegue < 14, "train", np.where(pliegue < 17, "val", "test"))
    df = df.copy()
    df["conjunto"] = conjunto
    return df


def verificar_sin_fuga(df: pd.DataFrame) -> dict:
    """CP06: ningún paciente aparece en dos conjuntos."""
    pacientes = {c: set(df.loc[df["conjunto"] == c, "patient_id"]) for c in ("train", "val", "test")}
    cruces = {
        "train_val": len(pacientes["train"] & pacientes["val"]),
        "train_test": len(pacientes["train"] & pacientes["test"]),
        "val_test": len(pacientes["val"] & pacientes["test"]),
    }
    return {"cruces": cruces, "sin_fuga": sum(cruces.values()) == 0,
            "pacientes": {c: len(p) for c, p in pacientes.items()},
            "imagenes": df["conjunto"].value_counts().to_dict()}


def cargar_imagenes(rutas) -> np.ndarray:
    """Lee y redimensiona con la misma función de la API (HU02). Devuelve uint8."""
    salida = np.zeros((len(rutas), 224, 224, 3), dtype=np.uint8)
    for i, r in enumerate(rutas):
        with Image.open(r) as img:
            salida[i] = redimensionar(ImageOps.exif_transpose(img))
    return salida


def crear_dataset(x_uint8, y, lote=32, aumentar=False, semilla=42):
    import tensorflow as tf

    aumento = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=semilla),
        tf.keras.layers.RandomRotation(0.1, seed=semilla),
        tf.keras.layers.RandomZoom(0.1, seed=semilla),
        tf.keras.layers.RandomContrast(0.1, seed=semilla),
        tf.keras.layers.RandomBrightness(0.1, value_range=(0, 255), seed=semilla),
    ])
    y_cat = tf.keras.utils.to_categorical(y, num_classes=len(CLASES))
    ds = tf.data.Dataset.from_tensor_slices((x_uint8, y_cat))
    if aumentar:
        ds = ds.shuffle(len(y), seed=semilla, reshuffle_each_iteration=True)
    ds = ds.batch(lote)

    def preparar(x, t):
        x = tf.cast(x, tf.float32)
        if aumentar:
            x = aumento(x, training=True)
            x = tf.clip_by_value(x, 0.0, 255.0)
        return normalizar(x), t

    return ds.map(preparar, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
