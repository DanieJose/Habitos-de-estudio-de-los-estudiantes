import streamlit as st
import pandas as pd
from src import data_manager, analysis, visualization
import base64

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Dashboard de Hábitos de Estudio", layout="wide", initial_sidebar_state="expanded")

# --- ESTADO DE LA SESIÓN ---
# Inicializa el estado de la sesión para mantener la información entre interacciones
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.df_original = None
    st.session_state.df_edited = None
    st.session_state.artefactos = None

# --- CARGA DE DATOS ---
@st.cache_data
def cargar_datos_iniciales(archivo_subido):
    """Carga los datos desde el archivo subido por el usuario o desde la ruta por defecto."""
    df_source = archivo_subido if archivo_subido else 'data/segmentacion_estudiantes_habitos.csv'
    df = data_manager.cargar_csv(df_source)
    if df is not None:
        try:
            data_manager.validar_datos(df)
            return df
        except ValueError as e:
            st.error(f"Error en el archivo CSV: {e}")
            return None
    return None

# --- FUNCIÓN PARA DESCARGAR DATOS ---
def get_table_download_link(df, filename, text):
    """Genera un enlace para descargar un DataFrame como archivo CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.title("⚙️ Panel de Control")
    archivo_subido = st.file_uploader("Carga tus propios datos (CSV)", type=['csv'])
    
    df_inicial = cargar_datos_iniciales(archivo_subido)
    
    if df_inicial is not None:
        # Si se carga un nuevo archivo o es la primera vez, se resetean los dataframes
        if st.session_state.df_original is None or archivo_subido is not None:
            st.session_state.df_original = df_inicial.copy()
            st.session_state.df_edited = df_inicial.copy()

        st.header("Parámetros del Análisis")
        num_clusters = st.slider("Número de clusters (k)", 2, 8, 3, help="Define en cuántos grupos se segmentarán los estudiantes.")
        
        if st.button("🚀 Ejecutar Análisis", type="primary", use_container_width=True):
            df_para_analisis = st.session_state.df_edited
            with st.spinner('Procesando datos y entrenando modelos...'):
                _, X_pca, df_encoded, _ = data_manager.preparar_datos(df_para_analisis)
                clusters, artefactos = analysis.realizar_analisis_completo(df_encoded, k=num_clusters, forzar_entrenamiento=True)
                df_encoded['Cluster'] = clusters
                
                # Guardar los resultados en el estado de la sesión
                st.session_state.analysis_done = True
                st.session_state.df_processed = df_encoded
                st.session_state.X_pca = X_pca
                st.session_state.perfiles, st.session_state.interpretaciones = analysis.interpretar_clusters(df_encoded)
                st.session_state.artefactos = artefactos
            st.success("¡Análisis completado!")

# --- PÁGINA PRINCIPAL ---
st.title("👨‍🎓️ Dashboard Estratégico de Hábitos de Estudio")

if df_inicial is None:
    st.error("No se pudieron cargar los datos. Por favor, asegúrate de que el archivo 'data/segmentacion_estudiantes_habitos.csv' exista o sube un nuevo archivo.")
else:
    # Definición de las pestañas de la aplicación
    tabs = st.tabs([
        "📝 Editor de Datos", "📊 Análisis Exploratorio", 
        "📈 Segmentación y Perfiles", "💡 Simulación y Predicción", "🧑‍🎓 Ficha del Estudiante"
    ])

    # Pestaña 0: Editor de Datos
    with tabs[0]:
        st.header("Editor Interactivo de Datos")
        st.info("Puedes editar, añadir o eliminar filas. Los cambios son temporales para esta sesión y no modificarán el archivo original.", icon="ℹ️")
        st.session_state.df_edited = st.data_editor(st.session_state.df_original, num_rows="dynamic", use_container_width=True)

    # Pestaña 1: Análisis Exploratorio
    with tabs[1]:
        st.header("Análisis Descriptivo y Exploratorio")
        df_exploratorio = st.session_state.df_edited
        
        st.subheader("Vista Panorámica de Relaciones (Pair Plot)")
        if st.session_state.analysis_done:
            # Si el análisis se ha hecho, usamos el dataframe procesado para colorear por cluster
            pairplot_df = st.session_state.df_processed.copy()
            fig_pair = visualization.plot_pairplot(pairplot_df)
        else:
            fig_pair = visualization.plot_pairplot(df_exploratorio, cluster_column=None)
        st.pyplot(fig_pair)
        
        col1, col2 = st.columns(2)
        with col1:
            var_hist = st.selectbox("Distribución de Variable:", options=['Calificacion_promedio', 'Horas_de_estudio_por_semana'])
            st.plotly_chart(visualization.plot_dynamic_histogram(df_exploratorio, var_hist), use_container_width=True)
        with col2:
            st.subheader("Correlación entre Variables Numéricas")
            # Usamos el dataframe procesado si está disponible para incluir las variables codificadas
            df_corr = st.session_state.df_processed if st.session_state.analysis_done else df_exploratorio
            st.plotly_chart(visualization.plot_correlation_heatmap(df_corr), use_container_width=True)

    # Las pestañas 2, 3 y 4 solo se muestran si el análisis se ha completado
    if st.session_state.analysis_done:
        # Pestaña 2: Segmentación y Perfiles
        with tabs[2]:
            st.header("Resultados de la Segmentación")
            col1, col2 = st.columns([0.6, 0.4])
            with col1:
                st.plotly_chart(visualization.plot_clusters_pca_interactive(st.session_state.X_pca, st.session_state.df_processed), use_container_width=True)
            with col2:
                st.plotly_chart(visualization.plot_radar_chart(st.session_state.perfiles), use_container_width=True)
            
            st.subheader("Explorar Datos por Cluster")
            cluster_seleccionado = st.selectbox("Selecciona un clúster para ver detalles:", st.session_state.perfiles.index)
            
            # Mostrar la interpretación y recomendación del clúster
            interpretacion_cluster = next((item for item in st.session_state.interpretaciones if item["cluster"] == cluster_seleccionado), None)
            if interpretacion_cluster:
                st.info(f"**Perfil:** {interpretacion_cluster['tipo']}\n\n**Recomendación:** {interpretacion_cluster['recomendacion']}")

            df_filtrado = st.session_state.df_processed[st.session_state.df_processed['Cluster'] == cluster_seleccionado]
            st.dataframe(df_filtrado)
            st.markdown(get_table_download_link(df_filtrado, f"cluster_{cluster_seleccionado}_data.csv", "📥 Descargar datos del clúster"), unsafe_allow_html=True)

        # Pestaña 3: Simulación y Predicción
        with tabs[3]:
            st.header("Análisis Predictivo y Simulación de Escenarios")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Factores Clave en la Calificación")
                fig_importance = visualization.plot_feature_importance(
                    st.session_state.artefactos['modelo_regresion'],
                    st.session_state.artefactos['features_pred']
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            with col2:
                st.subheader("Simulador de Escenarios 'What-If'")
                with st.container(border=True):
                    sim_horas = st.slider("Horas de estudio por semana (Sim)", 0, 40, 10)
                    sim_horario_str = st.selectbox("Horario preferido (Sim)", ["Mañana", "Tarde", "Noche"])
                    sim_espacio_str = st.radio("¿Tiene espacio de estudio? (Sim)", ["No", "Sí"], horizontal=True)
                    sim_repaso_str = st.select_slider("Frecuencia de repaso (Sim)", options=['Nunca', 'Rara vez', 'A veces', 'Frecuentemente', 'Siempre'])
                    sim_tecnicas_str = st.radio("¿Usa técnicas de estudio? (Sim)", ["No", "Sí"], horizontal=True)

                    # Crear DataFrame para la simulación con los valores codificados
                    datos_simulados = pd.DataFrame({
                        'Horas_de_estudio_por_semana': [sim_horas],
                        'Horario_preferido': [{'Mañana': 0, 'Tarde': 1, 'Noche': 2}[sim_horario_str]],
                        'Tiene_espacio_estudio': [{'No': 0, 'Sí': 1}[sim_espacio_str]],
                        'Frecuencia_repaso_apuntes': [{'Nunca': 0, 'Rara vez': 1, 'A veces': 2, 'Frecuentemente': 3, 'Siempre': 4}[sim_repaso_str]],
                        'Usa_tecnicas_estudio': [{'No': 0, 'Sí': 1}[sim_tecnicas_str]]
                    })
                    calificacion_predicha = st.session_state.artefactos['modelo_regresion'].predict(datos_simulados)[0]
                    st.metric(label="Calificación Predicha", value=f"{calificacion_predicha:.2f} / 100")
        
        # Pestaña 4: Ficha del Estudiante
        with tabs[4]:
            st.header("Perfil Individual del Estudiante")
            student_id = st.selectbox("Selecciona un ID de Estudiante:", options=st.session_state.df_processed['ID'].unique())
            
            if student_id:
                student_data = st.session_state.df_processed[st.session_state.df_processed['ID'] == student_id]
                cluster_id = student_data['Cluster'].iloc[0]
                cluster_avg = st.session_state.perfiles.loc[cluster_id]
                
                interpretacion = next(item for item in st.session_state.interpretaciones if item["cluster"] == cluster_id)
                st.info(f"**Este estudiante pertenece al Cluster {cluster_id}: {interpretacion['tipo']}**")
                
                fig_card = visualization.plot_student_profile_card(student_data, cluster_avg)
                st.pyplot(fig_card)
    else:
        st.info("🚀 Ejecuta el análisis en el panel de control para ver la segmentación, predicciones y fichas de estudiantes.")