import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_clusters_pca_interactive(X_pca, df_con_clusters):
    """Visualiza los clusters de estudiantes de forma interactiva usando Plotly."""
    df_plot = df_con_clusters.copy()
    df_plot['PC1'] = X_pca[:, 0]
    df_plot['PC2'] = X_pca[:, 1]
    df_plot['Cluster'] = df_plot['Cluster'].astype(str)
    
    fig = px.scatter(
        df_plot, x='PC1', y='PC2', color='Cluster',
        title="Segmentación de Estudiantes (Visualización con PCA)",
        hover_data=['ID', 'Calificacion_promedio', 'Horas_de_estudio_por_semana'],
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={'PC1': 'Componente Principal 1', 'PC2': 'Componente Principal 2'}
    )
    fig.update_layout(legend_title_text='Cluster')
    return fig

def plot_correlation_heatmap(df):
    """Genera un mapa de calor interactivo de correlaciones."""
    df_numeric = df.select_dtypes(include=['number'])
    corr = df_numeric.corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, aspect="auto", color_continuous_scale='viridis',
        title="Mapa de Calor de Correlaciones"
    )
    return fig

def plot_dynamic_histogram(df, column):
    """Genera un histograma interactivo para una columna seleccionada."""
    fig = px.histogram(
        df, x=column, title=f"Distribución de: {column.replace('_', ' ').capitalize()}",
        labels={column: column.replace('_', ' ').capitalize()}, color_discrete_sequence=['#007BFF']
    )
    return fig

def plot_radar_chart(perfiles):
    """Crea un gráfico radar interactivo para comparar los perfiles de los clusters."""
    from sklearn.preprocessing import MinMaxScaler
    
    # Usar todas las características numéricas para el radar
    radar_features = perfiles.columns.tolist()
    perfiles_radar = perfiles[radar_features]
    
    scaler = MinMaxScaler()
    perfiles_scaled = pd.DataFrame(scaler.fit_transform(perfiles_radar), columns=perfiles_radar.columns, index=perfiles_radar.index)
    
    fig = go.Figure()
    for i in perfiles_scaled.index:
        fig.add_trace(go.Scatterpolar(
            r=perfiles_scaled.loc[i].values,
            theta=[col.replace('_', ' ').capitalize() for col in perfiles_scaled.columns],
            fill='toself', name=f'Cluster {i}'
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, title="Gráfico Radar Comparativo de Perfiles"
    )
    return fig

def plot_pairplot(df, cluster_column='Cluster'):
    """Genera un gráfico de dispersión por pares (pair plot) y lo devuelve como una figura."""
    if cluster_column not in df.columns or df[cluster_column].isnull().all():
        fig = sns.pairplot(df.select_dtypes(include=['number']), diag_kind='kde')
    else:
        fig = sns.pairplot(df, hue=cluster_column, palette='viridis', diag_kind='kde')
    
    fig.fig.suptitle("Análisis Bivariado de Características", y=1.02)
    return fig

def plot_feature_importance(modelo_regresion, feature_names):
    """Crea un gráfico de barras interactivo con la importancia de cada factor."""
    importances = modelo_regresion.feature_importances_
    feature_importance_df = pd.DataFrame({'Factor': feature_names, 'Importancia': importances}).sort_values(by='Importancia', ascending=False)
    
    # Capitalizar y reemplazar guiones bajos para mejor legibilidad
    feature_importance_df['Factor'] = feature_importance_df['Factor'].str.replace('_', ' ').str.capitalize()
    
    fig = px.bar(
        feature_importance_df, x='Importancia', y='Factor', orientation='h',
        title="Importancia de cada Hábito en la Calificación",
        labels={'Importancia': 'Nivel de Impacto', 'Factor': 'Hábito de Estudio'},
        text_auto='.2f'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def plot_student_profile_card(student_data, cluster_avg):
    """
    Crea una "ficha" visual que compara los hábitos de un estudiante
    con el promedio de su cluster usando Matplotlib.
    """
    features = [
        'Horas_de_estudio_por_semana', 'Calificacion_promedio',
        'Horario_preferido', 'Tiene_espacio_estudio',
        'Frecuencia_repaso_apuntes', 'Usa_tecnicas_estudio'
    ]
    
    student_values = student_data[features].values.flatten()
    avg_values = cluster_avg.reindex(features).values.flatten()

    student_display = student_values.copy().astype(object)
    
    # Índices de las características para un manejo más fácil
    horario_idx = features.index('Horario_preferido')
    espacio_idx = features.index('Tiene_espacio_estudio')
    repaso_idx = features.index('Frecuencia_repaso_apuntes')
    tecnicas_idx = features.index('Usa_tecnicas_estudio')

    # Decodificar valores para mostrarlos en el gráfico
    student_display[horario_idx] = {0: 'Mañana', 1: 'Tarde', 2: 'Noche'}.get(student_values[horario_idx], 'N/A')
    student_display[espacio_idx] = {0: 'No', 1: 'Sí'}.get(student_values[espacio_idx], 'N/A')
    student_display[repaso_idx] = {0: 'Nunca', 1: 'Rara vez', 2: 'A veces', 3: 'Frec.', 4: 'Siempre'}.get(student_values[repaso_idx], 'N/A')
    student_display[tecnicas_idx] = {0: 'No', 1: 'Sí'}.get(student_values[tecnicas_idx], 'N/A')

    fig, ax = plt.subplots(figsize=(10, 6))

    # Separar características numéricas de categóricas para graficarlas de forma diferente
    numeric_features = ['Horas_de_estudio_por_semana', 'Calificacion_promedio']
    categorical_features = ['Horario_preferido', 'Tiene_espacio_estudio', 'Frecuencia_repaso_apuntes', 'Usa_tecnicas_estudio']

    numeric_indices = [features.index(f) for f in numeric_features]
    categorical_indices = [features.index(f) for f in categorical_features]
    
    # Graficar barras para las características numéricas
    ax.barh([p - 0.2 for p in numeric_indices], student_values[numeric_indices], height=0.4, label='Estudiante', color='skyblue', alpha=0.8)
    ax.barh([p + 0.2 for p in numeric_indices], avg_values[numeric_indices], height=0.4, label='Promedio del Cluster', color='orange', alpha=0.8)
    
    # Mostrar texto para las características categóricas
    for i in categorical_indices:
        ax.text(0.02, i, f"Estudiante: {student_display[i]}", ha='left', va='center', fontsize=11,
                bbox=dict(facecolor='skyblue', alpha=0.5, boxstyle='round,pad=0.3'), transform=ax.get_yaxis_transform())
        ax.axhspan(i - 0.45, i + 0.45, color='whitesmoke', zorder=0)

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([f.replace('_', ' ').capitalize() for f in features])
    ax.invert_yaxis()
    ax.set_xlabel('Valor / Calificación')
    ax.set_title(f'Perfil del Estudiante {student_data["ID"].iloc[0]} vs. Promedio del Cluster {student_data["Cluster"].iloc[0]}')
    ax.legend()
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    return fig