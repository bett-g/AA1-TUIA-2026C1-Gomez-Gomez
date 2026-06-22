# MLOps Deployment - Clasificación de Lluvia (Docker)

Este directorio contiene todo lo necesario para realizar inferencias sobre nuevos datos climáticos utilizando el modelo de **Regresión Logística Optimizado** entrenado en el Trabajo Práctico. Se proveen instrucciones tanto para ejecutarlo de forma aislada mediante **Docker** como de forma local directa con **Python**.

## Contenido de la Carpeta

*   `inferencia.py`: Script de Python para cargar el modelo y preprocesar los nuevos datos en lote, generando las predicciones.
*   `requirements.txt`: Archivo con las dependencias mínimas indispensables para realizar la inferencia (`pandas`, `numpy`, `scikit-learn`, `joblib`).
*   `Dockerfile`: Configuración para empaquetar el modelo y su entorno.
*   `*.joblib`: Archivos binarios serializados que contienen:
    *   `model_lr.joblib`: Pesos del modelo entrenado.
    *   `scaler.joblib`: Escalador robusto (`RobustScaler`) ajustado en entrenamiento.
    *   `encoder_zona.joblib`: Codificador de una sola columna (`OneHotEncoder`) ajustado para las localizaciones.
    *   `imputation_data.joblib`: Diccionario de metadatos con las estadísticas de imputación deterministas (medianas y modas grupales por `Zona` y `Estacion`).

---

## Opción A: Ejecución con Docker

Sigue estos pasos para construir y ejecutar el contenedor. Se asume que tienes Docker instalado y ejecutándose en tu máquina.

### 1. Construir la Imagen de Docker
Abre una terminal, navega hasta la carpeta `docker` y ejecuta:
```bash
docker build -t weather-inference:latest .
```

### 2. Ejecutar la Inferencia
Para pasar archivos CSV locales al contenedor y recibir las predicciones, debes montar un volumen utilizando la bandera `-v`.

El contenedor acepta dos argumentos obligatorios de línea de comandos:
1.  La ruta del archivo CSV de entrada (dentro del contenedor).
2.  La ruta del archivo CSV de salida (dentro del contenedor).

*   **En Windows (PowerShell):**
    ```powershell
    docker run --rm -v "C:\Ruta\A\Tus\Datos:/data" weather-inference:latest /data/datos_nuevos.csv /data/predicciones.csv
    ```
*   **En Windows (Command Prompt - CMD):**
    ```cmd
    docker run --rm -v "C:\Ruta\A\Tus\Datos:/data" weather-inference:latest /data/datos_nuevos.csv /data/predicciones.csv
    ```
*   **En macOS / Linux / Git Bash:**
    ```bash
    docker run --rm -v "/absolute/path/to/data:/data" weather-inference:latest /data/datos_nuevos.csv /data/predicciones.csv
    ```

> *Nota:* Reemplaza `C:\Ruta\A\Tus\Datos` o `/absolute/path/to/data` por la carpeta real en tu computadora donde se encuentra el archivo `datos_nuevos.csv`. El archivo de salida `predicciones.csv` se creará en esa misma carpeta.

---

## Opción B: Ejecución Alternativa SIN Docker (Python Local)

Si no tienes Docker instalado, puedes ejecutar el script de inferencia directamente utilizando el entorno de Python de tu sistema local.

### 1. Instalar las dependencias
Desde la terminal, instala las librerías necesarias ejecutando:
```bash
pip install pandas numpy scikit-learn==1.8.0 joblib
```
*(Nota: Se recomienda utilizar la versión `1.8.0` de `scikit-learn` para asegurar la correcta des-serialización de los archivos `.joblib` entrenados).*

### 2. Ejecutar la Inferencia
Abre una terminal en la carpeta raíz del proyecto y corre el script pasándole el archivo de entrada y de salida como argumentos:

*   **Si ejecutas desde la carpeta raíz del repositorio:**
    ```bash
    python docker/inferencia.py weatherAUS_2026C1.csv resultado_local.csv
    ```
*   **Si ejecutas desde adentro de la carpeta `docker/`:**
    ```bash
    python inferencia.py ../weatherAUS_2026C1.csv ../resultado_local.csv
    ```

---

## Estructura del Archivo de Entrada y Salida

*   **Archivo de entrada (`datos_nuevos.csv`):** Debe ser un archivo CSV con las mismas columnas climáticas del dataset original de entrenamiento (pueden venir nulas y el pipeline las imputará automáticamente). No es necesario incluir `RainTomorrow` ni `Unnamed: 0`.
*   **Archivo de salida (`predicciones.csv`):** Contendrá las mismas filas y columnas del archivo original de entrada, y se le anexarán dos columnas adicionales al final:
    *   `RainTomorrow_pred`: Predicción de la clase de lluvia de mañana ("No" o "Yes").
    *   `RainTomorrow_prob`: Probabilidad estimada de lluvia para el día de mañana (valor entre 0.0 y 1.0).
