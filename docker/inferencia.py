import pandas as pd
import numpy as np
import joblib
import argparse
import sys
import os

COLS_ORDENADAS = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed',
    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
    'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday',
    'Month', 'WindGustDir_sin', 'WindGustDir_cos', 'WindDir9am_sin',
    'WindDir9am_cos', 'WindDir3pm_sin', 'WindDir3pm_cos', 'Estacion_sin',
    'Estacion_cos', 'Zona_Subtropical', 'Zona_Templada-SE', 'Zona_Tropical', 'Zona_Árida'
]

COLS_NUMERICAS = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed',
    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
    'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'
]

def load_artifacts(docker_dir):
    """Carga el modelo, scaler, encoder e imputation metadata desde la carpeta especificada."""
    try:
        model = joblib.load(os.path.join(docker_dir, 'model_lr.joblib'))
        scaler = joblib.load(os.path.join(docker_dir, 'scaler.joblib'))
        encoder_zona = joblib.load(os.path.join(docker_dir, 'encoder_zona.joblib'))
        imputation_data = joblib.load(os.path.join(docker_dir, 'imputation_data.joblib'))
        return model, scaler, encoder_zona, imputation_data
    except Exception as e:
        print(f"Error al cargar los artefactos del modelo en '{docker_dir}': {e}")
        sys.exit(1)

def preprocesar_datos(df, imp_data, encoder_zona, scaler):
    """Aplica la transformación y limpieza de datos exacta al DataFrame de entrada."""
    df_proc = df.copy()
    
    df_proc = df_proc.drop(columns=['Unnamed: 0', 'RainTomorrow', 'RainfallTomorrow'], errors='ignore')
    
    if 'Date' in df_proc.columns:
        df_proc['Date'] = pd.to_datetime(df_proc['Date'], errors='coerce')
        df_proc['Month'] = df_proc['Date'].dt.month
        df_proc.drop('Date', axis=1, inplace=True)
    elif 'Month' not in df_proc.columns:
        df_proc['Month'] = 6
        
    df_proc.drop(columns=['Year', 'Day'], inplace=True, errors='ignore')
    
    if 'Location' in df_proc.columns:
        df_proc['Zona'] = df_proc['Location'].map(imp_data['zone_map'])
        df_proc.drop('Location', axis=1, inplace=True)
    else:
        df_proc['Zona'] = np.nan
        
    df_proc['Estacion'] = df_proc['Month'].map(imp_data['mapeo_estaciones'])
    
    df_proc['Zona'] = df_proc['Zona'].fillna('Templada-SE') # Zona más común en train
    df_proc['Estacion'] = df_proc['Estacion'].fillna('Invierno') # Estación más común en train

    for col in imp_data['cols_to_impute_median']:
        if col not in df_proc.columns:
            df_proc[col] = np.nan
        group_keys = df_proc.set_index(['Zona', 'Estacion']).index
        group_vals = group_keys.map(imp_data['numerical_medians_grouped'][col])
        df_proc[col] = df_proc[col].fillna(pd.Series(group_vals, index=df_proc.index)).fillna(imp_data['numerical_medians_global'][col])
        
    for col in imp_data['cols_to_impute_clouds']:
        if col not in df_proc.columns:
            df_proc[col] = np.nan
        group_keys = df_proc.set_index(['Zona', 'Estacion']).index
        group_vals = group_keys.map(imp_data['clouds_medians_grouped'][col])
        df_proc[col] = df_proc[col].fillna(pd.Series(group_vals, index=df_proc.index)).fillna(imp_data['clouds_medians_global'][col])
        # Redondear y clip
        df_proc[col] = df_proc[col].astype(float).round().clip(0, 8).fillna(4).astype('Int64')
        
    for col in imp_data['cols_to_impute_mode']:
        if col not in df_proc.columns:
            df_proc[col] = np.nan
        group_keys = df_proc.set_index(['Zona', 'Estacion']).index
        group_vals = group_keys.map(imp_data['cat_modes_grouped'][col])
        df_proc[col] = df_proc[col].fillna(pd.Series(group_vals, index=df_proc.index)).fillna(imp_data['cat_modes_global'][col])

    columnas_viento = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
    for col in columnas_viento:
        if col in df_proc.columns:
            radianes = df_proc[col].map(imp_data['dir_mapping']).fillna(0) * (np.pi / 180)
            df_proc[f'{col}_sin'] = np.sin(radianes)
            df_proc[f'{col}_cos'] = np.cos(radianes)
            df_proc.drop(columns=[col], inplace=True)
            
    if 'Estacion' in df_proc.columns:
        radianes_est = df_proc['Estacion'].map(imp_data['estacion_mapping']).fillna(0) * (np.pi / 180)
        df_proc['Estacion_sin'] = np.sin(radianes_est)
        df_proc['Estacion_cos'] = np.cos(radianes_est)
        df_proc.drop(columns=['Estacion'], inplace=True)
        
    df_proc['RainToday'] = df_proc['RainToday'].map(imp_data['mapeo_binario']).fillna(0).astype(int)

    zonas_encoded = encoder_zona.transform(df_proc[['Zona']])
    df_proc = pd.concat([df_proc.drop(columns=['Zona']), zonas_encoded], axis=1)

    for col in COLS_ORDENADAS:
        if col not in df_proc.columns:
            df_proc[col] = 0.0
            
    df_proc = df_proc[COLS_ORDENADAS]

    df_proc[COLS_NUMERICAS] = scaler.transform(df_proc[COLS_NUMERICAS])
    
    return df_proc

def main():
    parser = argparse.ArgumentParser(description="Script de inferencia MLOps para predicción de lluvia mañana.")
    parser.add_argument("input", help="Ruta al archivo CSV de entrada con los datos climáticos")
    parser.add_argument("output", help="Ruta al archivo CSV de salida donde se guardarán las predicciones")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Cargando artefactos del modelo serializado...")
    model, scaler, encoder_zona, imputation_data = load_artifacts(script_dir)
    
    print(f"Cargando datos de entrada desde: {args.input} ...")
    try:
        df_original = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error al leer el archivo de entrada '{args.input}': {e}")
        sys.exit(1)
        
    if len(df_original) == 0:
        print("El archivo de entrada está vacío.")
        sys.exit(1)
        
    print("Preprocesando datos...")
    df_preprocesado = preprocesar_datos(df_original, imputation_data, encoder_zona, scaler)
    
    print("Realizando inferencia...")
    X_inference = df_preprocesado.astype(float)
    
    probabilidades = model.predict_proba(X_inference)[:, 1]
    predicciones = (probabilidades >= 0.5).astype(int)
    
    pred_labels = pd.Series(predicciones).map({0: 'No', 1: 'Yes'})
    
    df_resultado = df_original.copy()
    df_resultado['RainTomorrow_pred'] = pred_labels
    df_resultado['RainTomorrow_prob'] = np.round(probabilidades, 4)
    
    print(f"Guardando resultados en: {args.output} ...")
    try:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_resultado.to_csv(args.output, index=False)
        print("Inferencia finalizada exitosamente.")
    except Exception as e:
        print(f"Error al escribir el archivo de salida '{args.output}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
