import streamlit as st
import requests
import os

st.set_page_config(page_title="AcePoint Video Analysis", layout="centered")
st.title("🎾 AcePoint – Análisis de Vídeo de Pádel con IA")

API_URL = "https://[TU_REPLIT_URL].repl.co"  # Substitueix això pel teu enllaç Replit públic

# Subida de vídeo
st.header("📤 Subir vídeo para análisis")
uploaded_file = st.file_uploader("Selecciona un archivo de vídeo .mp4", type=["mp4"])

if uploaded_file is not None:
    if st.button("Enviar vídeo"):
        with st.spinner("Subiendo y procesando vídeo..."):
            files = {"video": uploaded_file.getvalue()}
            response = requests.post(f"{API_URL}/upload", files={"video": uploaded_file})
            if response.status_code == 200:
                video_id = response.json()["video_id"]
                st.success("✅ Vídeo procesado correctamente")
                st.code(f"ID del vídeo: {video_id}")

                # Obtener resultados
                result = requests.get(f"{API_URL}/result/{video_id}")
                if result.status_code == 200:
                    kpis = result.json()
                    st.header("📊 Resultados del análisis")
                    st.json(kpis)
                else:
                    st.warning("Procesado, pero sin resultados disponibles aún.")
            else:
                st.error("❌ Error al subir el vídeo.")