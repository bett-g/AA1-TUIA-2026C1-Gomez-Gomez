FROM python:3.11-slim

# Evitar que Python escriba archivos .pyc en disco y almacene en búfer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copiar requerimientos e instalar dependencias
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copiar script de inferencia y binarios del modelo
COPY inferencia.py /app/
COPY *.joblib /app/

# Punto de entrada predeterminado
ENTRYPOINT ["python", "inferencia.py"]
