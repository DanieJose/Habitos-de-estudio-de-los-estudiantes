import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def cargar_csv(ruta):
    """
    Carga un archivo CSV con manejo de excepciones.
    """
    try:
        return pd.read_csv(ruta)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: {ruta}")
        return None
    except Exception as e:
        print(f"Ocurrió un error inesperado al leer el archivo: {e}")
        return None

def validar_datos(df):
    """
    Valida la consistencia de los datos del DataFrame.
    """
    if not df['Calificacion_promedio'].between(0, 100).all():
        raise ValueError("Error de consistencia: Se encontraron calificaciones fuera del rango válido (0-100).")
    
    print("✅ Datos validados correctamente.")
    return True

def preparar_datos(df):
    """
    Codifica variables categóricas, normaliza los datos y aplica PCA.
    Devuelve los datos procesados, el objeto PCA, el DataFrame codificado y el scaler.
    """
    df_encoded = df.copy()

    # Codificación de variables categóricas
    df_encoded['Horario_preferido'] = df_encoded['Horario_preferido'].map({'Mañana': 0, 'Tarde': 1, 'Noche': 2})
    df_encoded['Tiene_espacio_estudio'] = df_encoded['Tiene_espacio_estudio'].map({'No': 0, 'Sí': 1})
    df_encoded['Frecuencia_repaso_apuntes'] = df_encoded['Frecuencia_repaso_apuntes'].map({
        'Nunca': 0, 'Rara vez': 1, 'A veces': 2, 'Frecuentemente': 3, 'Siempre': 4
    })
    df_encoded['Usa_tecnicas_estudio'] = df_encoded['Usa_tecnicas_estudio'].map({'No': 0, 'Sí': 1})

    features = [
        'Horas_de_estudio_por_semana', 'Horario_preferido', 'Tiene_espacio_estudio',
        'Frecuencia_repaso_apuntes', 'Usa_tecnicas_estudio', 'Calificacion_promedio'
    ]
    X = df_encoded[features]

    # Normalización de los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Análisis de Componentes Principales (PCA) para visualización
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print("✅ Datos preparados y normalizados.")
    # DEVOLVEMOS EL SCALER PARA PODER GUARDARLO
    return X_scaled, X_pca, df_encoded, scaler