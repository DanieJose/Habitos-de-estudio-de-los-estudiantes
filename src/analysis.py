import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def guardar_artefactos(artefactos, ruta='artefactos_analisis.pkl'):
    """Guarda un diccionario de artefactos (modelos, scaler, etc.) en un archivo pickle."""
    try:
        with open(ruta, 'wb') as f:
            pickle.dump(artefactos, f)
        print(f"✅ Artefactos guardados exitosamente en {ruta}")
    except Exception as e:
        print(f"Error al guardar los artefactos: {e}")

def cargar_artefactos(ruta='artefactos_analisis.pkl'):
    """Carga un diccionario de artefactos desde un archivo pickle."""
    try:
        with open(ruta, 'rb') as f:
            artefactos = pickle.load(f)
        print(f"✅ Artefactos cargados exitosamente desde {ruta}")
        return artefactos
    except FileNotFoundError:
        print(f"Advertencia: No se encontraron artefactos guardados en {ruta}. Se crearán nuevos.")
        return None
    except Exception as e:
        print(f"Error al cargar los artefactos: {e}")
        return None

def entrenar_modelo_predictivo(X, y):
    """Entrena un modelo RandomForestRegressor para predecir calificaciones."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regresor = RandomForestRegressor(n_estimators=100, random_state=42)
    regresor.fit(X_train, y_train)
    score = regresor.score(X_test, y_test)
    print(f"✅ Modelo predictivo entrenado con un R^2 score de: {score:.2f}")
    return regresor

def realizar_analisis_completo(df_encoded, k=3, forzar_entrenamiento=False):
    """
    Realiza el clustering y entrena el modelo predictivo.
    Gestiona el guardado y la carga de modelos (artefactos).
    """
    ruta_artefactos = 'artefactos_analisis.pkl'
    artefactos = None

    if not forzar_entrenamiento:
        artefactos = cargar_artefactos(ruta_artefactos)

    # Definición de las características para cada modelo
    features_cluster = ['Horas_de_estudio_por_semana', 'Horario_preferido', 'Tiene_espacio_estudio',
                        'Frecuencia_repaso_apuntes', 'Usa_tecnicas_estudio', 'Calificacion_promedio']
    features_pred = ['Horas_de_estudio_por_semana', 'Horario_preferido', 'Tiene_espacio_estudio',
                     'Frecuencia_repaso_apuntes', 'Usa_tecnicas_estudio']

    if artefactos is None:
        print("Entrenando nuevos modelos...")
        # Entrenar modelo de clustering (KMeans)
        scaler = StandardScaler()
        X_cluster_scaled = scaler.fit_transform(df_encoded[features_cluster])
        
        modelo_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        modelo_kmeans.fit(X_cluster_scaled)
        
        # Entrenar modelo de regresión para predicción
        X_pred_data = df_encoded[features_pred]
        y_pred_data = df_encoded['Calificacion_promedio']
        modelo_regresion = entrenar_modelo_predictivo(X_pred_data, y_pred_data)
        
        artefactos = {
            'modelo_kmeans': modelo_kmeans, 
            'scaler_cluster': scaler,
            'modelo_regresion': modelo_regresion,
            'features_cluster': features_cluster,
            'features_pred': features_pred
        }
        guardar_artefactos(artefactos, ruta_artefactos)
    
    # Usar los modelos (cargados o recién entrenados) para predecir
    X_cluster_scaled = artefactos['scaler_cluster'].transform(df_encoded[artefactos['features_cluster']])
    clusters = artefactos['modelo_kmeans'].predict(X_cluster_scaled)
    
    return clusters, artefactos

def interpretar_clusters(df_con_clusters):
    """Calcula las medias de las características para cada clúster y genera una interpretación textual."""
    perfiles = df_con_clusters.groupby('Cluster')[[
        "Horas_de_estudio_por_semana", "Calificacion_promedio",
        "Tiene_espacio_estudio", "Usa_tecnicas_estudio",
        "Horario_preferido", "Frecuencia_repaso_apuntes"
    ]].mean().round(2)

    interpretaciones = []
    for cluster_id, row in perfiles.iterrows():
        horas = row['Horas_de_estudio_por_semana']
        calificacion = row['Calificacion_promedio']
        
        # Lógica mejorada para la interpretación de perfiles
        if calificacion >= 85:
            tipo = "Estudiantes de Alto Rendimiento"
            recomendacion = "Fomentar su participación como mentores. Ofrecerles retos académicos avanzados."
        elif horas >= 15 and calificacion < 75:
            tipo = "Estudiantes Esforzados con Bajo Rendimiento"
            recomendacion = "Revisar y optimizar sus técnicas de estudio. Sugerir tutorías para identificar barreras de aprendizaje."
        elif horas < 8 and calificacion < 70:
            tipo = "Estudiantes con Posible Riesgo Académico"
            recomendacion = "Requieren intervención y seguimiento personalizado. Investigar posibles causas externas."
        else:
            tipo = "Estudiantes Promedio/Estándar"
            recomendacion = "Potencial de mejora con estrategias de gestión del tiempo y técnicas de estudio más activas."

        interpretaciones.append({
            "cluster": cluster_id,
            "perfil": perfiles.loc[cluster_id].to_dict(),
            "tipo": tipo,
            "recomendacion": recomendacion
        })
        
    return perfiles, interpretaciones