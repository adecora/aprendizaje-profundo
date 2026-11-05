import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_image(img, *, ax=None, title=None, **kwargs):
    """
    Función auxiliar para mostrar una imagen.

    Parámetros:
    ===========
    - img: La imagen a mostrar.
    - ax: Eje de Matplotlib donde se dibujará la imagen. Si es None, se creará uno nuevo.
    - title: Título opcional para la imagen.
    - **kwargs: Argumentos adicionales que se pasarán a ax.imshow().

    Retorna:
    ===========
    - ax: El eje de Matplotlib donde se dibujó la imagen.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img, **kwargs)
    if title is not None:
        ax.set_title(title)
    ax.axis("off")
    return ax


def summary_model(model, caption, label, *, bformat="kb"):
    """
    Función que resumen la arquitectur del modelo.
    Análoga a la función summary de Keras pero con salida formateada para Quarto.

    Parámetros:
    ===========
    - model: Modelo de Keras a resumir.
    - caption: Título de la tabla de resumen.
    - label: Etiqueta para referenciar la tabla en Quarto.
    - bformat: Formato de salida del tamaño de los parámetros (por defecto 'kb').

    Retorna:
    ===========
    - str: Cadena con el resumen del modelo en formato de tabla Markdown para Quarto.
    """
    dtype_bsize = {
        "uint8": 1,
        "uint16": 2,
        "float32": 4,
        "float64": 8,
    }
    bytes = ["b", "kb", "mb", "gb"]

    lines = [
        "| Capa (tipo) | Salida | Parámetros |",
        "|:------------|:-------|:-----------|",
    ]
    for layer in model.layers:
        name = layer.name
        type_ = type(layer).__name__
        output_shape = layer.output.shape
        params = layer.count_params()

        lines.append(f"| {name} ({type_}) | {output_shape} | {params:,} |")
    if caption or label:
        lines.append(f": {caption} {{{label} .striped .hover .sm}}")
    else:
        lines.append(": {.striped .hover .sm}")

    lines.append("")
    total_params = model.count_params()
    total_bytes = (
        total_params
        * dtype_bsize[model.compute_dtype]
        / (1024 ** bytes.index(bformat.lower()))
    )
    lines.append(
        f"**Total de parámetros:** {model.count_params():,} ({total_bytes:.2f} {bformat})"
    )
    return "\n".join(lines)


def summary_evaluation(E, caption, label):
    """
    Función que genera un resumen de la evaluación del modelo en formato de tabla Markdown para Quarto.

    Parámetros:
    ===========
    - E: Diccionario con los resultados de la evaluación (loss y accuracy), para cada modelo evaluado.
    - caption: Título de la tabla de resumen.
    - label: Etiqueta para referenciar la tabla en Quarto.

    Retorna:
    ===========
    - str: Cadena con el resumen de la evaluación en formato de tabla Markdown para Quarto.
    """
    df = pd.DataFrame(E, index=["loss", "accuracy"])
    df.loc["accuracy"] = (df.loc["accuracy"] * 100).round(2)
    lines = [
        f"{df.to_markdown()}",
        f": {caption} {{{label} .striped .hover .sm}}",
    ]
    return "\n".join(lines)


def pp_history(history, *, xlim=None, overfit=None):
    """
    Pretty print del historial de entrenamiento de los modelos.

    Parámetros:
    ===========
    - history: Lista de tuplas (nombre_modelo, historial), donde historial es un objeto History de Keras.
    - xlim: Lista de tuplas (xmin, xmax) para establecer los límites del eje x en cada gráfico.
    - overfit: Lista de enteros indicando la época en la que ocurre el sobreajuste para cada modelo.

    Retorna:
    ===========
    - fig: Figura de Matplotlib con los gráficos del historial de entrenamiento.
    - axes: Ejes de Matplotlib correspondientes a cada gráfico.
    """
    nrows = len(history)
    fig, axes = plt.subplots(1, nrows, figsize=(7 * nrows, 6))

    if xlim is None:
        xlim = [None] * nrows
    if overfit is None:
        overfit = [None] * nrows

    for idx, ax in enumerate(axes):
        H = history[idx][1]
        ax.plot(H.epoch, H.history["loss"], label="train_loss")
        ax.plot(H.epoch, H.history["val_loss"], label="val_loss")
        ax.plot(H.epoch, H.history["accuracy"], label="train_acc")
        ax.plot(H.epoch, H.history["val_accuracy"], label="val_acc")
        ax.legend()
        ax.set_title(f"Entrenamiento - {history[idx][0]}")
        ax.set_xlabel("Época")
        ax.set_ylabel("Pérdida/Precisión")

        if xlim[idx] is not None:
            ax.set_xlim(xlim[idx])

        if overfit[idx] is not None:
            ax.axvline(x=overfit[idx], color="orange", lw=2.5, ls="--", label="overfit")
            ax.legend()

    return fig, axes
