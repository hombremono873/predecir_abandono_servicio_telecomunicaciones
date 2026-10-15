# Predicción de abandono de clientes (Streamlit)

Aplicación web que usa el modelo entrenado en **abandon_clientes.ipynb** para predecir si un cliente cancelará el servicio.

## Archivos

| Archivo | Para qué sirve |
|---|---|
| app.py | La aplicación |
| iconos.py | Íconos de línea (Lucide, licencia libre ISC) incrustados como SVG |
| icono.png | Ícono de la pestaña del navegador |
| modelo_abandono.joblib | El modelo descargado desde la sección 5.1 del notebook |
| requirements.txt | Librerías necesarias |
| .streamlit/config.toml | Colores y tema de la aplicación |

## Cómo ejecutarla

El entorno virtual **.venv** ya está creado (Python 3.13) con todas las librerías instaladas. En PowerShell, desde esta carpeta:

    .venv\Scripts\Activate.ps1
    streamlit run app.py

Se abre el navegador en **http://localhost:8501**. Para detenerla: **Ctrl + C** en la terminal.

Si PowerShell bloquea la activación, ejecutar una sola vez:

    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

## Crear el entorno desde cero (otro computador)

1. Copia **modelo_abandono.joblib** en esta carpeta.
2. Crea el entorno e instala las librerías (scikit-learn debe ser la misma versión que imprime la celda 5.1; hoy es 1.6.1):

        py -3.13 -m venv .venv
        .venv\Scripts\Activate.ps1
        pip install -r requirements.txt

3. Ejecuta **streamlit run app.py**.

## Qué hace

- **Un cliente:** formulario con los datos del cliente. Muestra la predicción, la probabilidad de abandono, el nivel de riesgo y las variables que más pesaron en el resultado.
- **Varios clientes (CSV):** sube un archivo con una fila por cliente y descarga las predicciones. Hay una plantilla descargable dentro de la aplicación.

Si se vuelve a entrenar el modelo, solo hay que reemplazar **modelo_abandono.joblib**: el formulario muestra automáticamente las variables que el modelo usa.
