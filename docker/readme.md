# Trabajo Práctico - Clasificación - Aprendizaje Automático 1

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

## Instrucciones de Uso

Para poder construir y ejecutar el contenedor, sigue los siguientes pasos. Se asume que tienes Docker instalado y ejecutándose en tu máquina.

### 1. Construir la Imagen de Docker

Abre una terminal, navega hasta esta carpeta (`docker`) y ejecuta el siguiente comando para construir la imagen llamada `weather-inference`:

```bash
docker build -t weather-inference:latest .
```

### 2. Ejecutar la Inferencia con el Contenedor

Para poder pasar un archivo CSV local al contenedor y recibir las predicciones, debes montar un volumen utilizando la bandera `-v`. 

El contenedor acepta dos argumentos obligatorios de línea de comandos:
1.  La ruta del archivo CSV de entrada (dentro del contenedor).
2.  La ruta del archivo CSV de salida (dentro del contenedor).

#### Comando de Ejecución (Windows Powershell):

```powershell
docker run --rm -v "C:\Ruta\A\Tus\Datos:/data" weather-inference:latest /data/datos_nuevos.csv /data/predicciones.csv
```

#### Comando de Ejecución (macOS / Linux / Git Bash):

```bash
docker run --rm -v "/absolute/path/to/data:/data" weather-inference:latest /data/datos_nuevos.csv /data/predicciones.csv
```

> **Nota:** Reemplaza `C:\Ruta\A\Tus\Datos` o `/absolute/path/to/data` por la carpeta real en tu computadora donde se encuentra el archivo `datos_nuevos.csv`. El archivo de salida `predicciones.csv` se creará en esa misma carpeta.

---

## Estructura del Archivo de Entrada y Salida

*   **Archivo de entrada (`datos_nuevos.csv`):** Debe ser un archivo CSV con las mismas columnas climáticas del dataset original de entrenamiento (pueden venir nulas y el pipeline las imputará automáticamente). No es necesario incluir `RainTomorrow` ni `Unnamed: 0`.
*   **Archivo de salida (`predicciones.csv`):** Contendrá las mismas filas y columnas del archivo original de entrada, y se le anexarán dos columnas adicionales al final:
    *   `RainTomorrow_pred`: Predicción de la clase de lluvia de mañana ("No" o "Yes").
    *   `RainTomorrow_prob`: Probabilidad estimada de lluvia para el día de mañana (valor entre 0.0 y 1.0).
