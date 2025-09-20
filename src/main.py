import argparse
import data_manager
import analysis
import visualization

def main(forzar_entrenamiento):
    """
    Función principal que orquesta el análisis de hábitos de estudio.
    """
    # 1. Cargar datos
    ruta_csv = 'data/segmentacion_estudiantes_habitos.csv'
    df_original = data_manager.cargar_csv(ruta_csv)
    
    if df_original is None:
        return # Termina la ejecución si el archivo no se pudo cargar

    # 2. Validar y preparar datos
    try:
        data_manager.validar_datos(df_original)
        X_scaled, X_pca, df_encoded = data_manager.preparar_datos(df_original)
    except ValueError as e:
        print(e) # Imprime el mensaje de error de validación
        return

    # (Opcional) Visualizar el método del codo para elegir 'k'
    # visualization.plot_codo(X_scaled)
    
    # 3. Realizar clustering
    clusters, modelo = analysis.realizar_clustering(
        X_scaled, 
        k=3, 
        forzar_entrenamiento=forzar_entrenamiento
    )
    df_encoded['Cluster'] = clusters
    
    # 4. Interpretar resultados
    analysis.interpretar_clusters(df_encoded)

    # 5. Visualizar clusters
    visualization.plot_clusters_pca(X_pca, df_encoded)
    
if __name__ == '__main__':
    # Configuración para aceptar argumentos desde la línea de comandos
    parser = argparse.ArgumentParser(description="Análisis y segmentación de hábitos de estudio.")
    parser.add_argument(
        '--entrenar',
        action='store_true',
        help="Fuerza el reentrenamiento del modelo K-Means en lugar de cargar uno existente."
    )
    
    args = parser.parse_args()
    
    main(forzar_entrenamiento=args.entrenar)