"""Construcción de MobileNetV2 (arquitectura del producto) y MobileNetV3-Large (referencia, HU16)."""
from app.config import CLASES


def construir_modelo(arquitectura: str = "MobileNetV2", pesos: str | None = "imagenet", dropout: float = 0.3):
    """Devuelve (modelo, base, capa_gradcam). La base queda congelada (fase 1)."""
    import keras

    entradas = keras.Input(shape=(224, 224, 3), name="imagen")
    if arquitectura == "MobileNetV2":
        base = keras.applications.MobileNetV2(include_top=False, weights=pesos, input_tensor=entradas)
        capa_gradcam = "out_relu"
    elif arquitectura == "MobileNetV3Large":
        # Sin la capa interna de preprocesamiento: recibe la misma entrada en [-1, 1]
        base = keras.applications.MobileNetV3Large(include_top=False, weights=pesos, input_tensor=entradas,
                                                   include_preprocessing=False)
        capa_gradcam = [c.name for c in base.layers if len(c.output.shape) == 4][-1]
    else:
        raise ValueError(f"Arquitectura no soportada: {arquitectura}")

    base.trainable = False
    x = keras.layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    salidas = keras.layers.Dense(len(CLASES), activation="softmax", name="clasificacion")(x)
    modelo = keras.Model(entradas, salidas, name=arquitectura)
    return modelo, base, capa_gradcam


def descongelar_ultimas(base, n_capas: int) -> None:
    """Fase 2: ajusta las últimas n capas de la base, sin tocar las de BatchNormalization."""
    import keras

    base.trainable = True
    for capa in base.layers[:-n_capas]:
        capa.trainable = False
    for capa in base.layers[-n_capas:]:
        if isinstance(capa, keras.layers.BatchNormalization):
            capa.trainable = False
