# padel_ai_app.py
import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
from PIL import Image

st.set_page_config(page_title="Análisis IA Padel", layout="wide")
st.title("🏓 Análisis Inteligente de Pádel con IA")

st.markdown("Sube un video o imagen de un jugador y obtén análisis técnico y postural en tiempo real.")

# Cargar archivo multimedia
upload_option = st.radio("Selecciona el tipo de archivo", ["Imagen", "Video"], horizontal=True)

uploaded_file = st.file_uploader("Sube una imagen o video", type=["jpg", "png", "mp4"])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# Función para analizar postura
def analizar_pose(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return frame, result

if uploaded_file:
    if upload_option == "Imagen":
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        frame, result = analizar_pose(frame)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Análisis Postural", use_column_width=True)

        if result.pose_landmarks:
            st.success("Postura detectada correctamente")
            st.write("Landmarks clave:")
            for i, landmark in enumerate(result.pose_landmarks.landmark):
                st.write(f"{i}: x={landmark.x:.2f}, y={landmark.y:.2f}, z={landmark.z:.2f}")
        else:
            st.warning("No se detectó postura. Asegúrate de que el jugador esté claramente visible.")

    elif upload_option == "Video":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        frame_count = 0
        keypoints_summary = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > 100:  # limitar a 100 frames
                break
            frame, result = analizar_pose(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)
            frame_count += 1

        cap.release()
        st.success(f"Procesamiento finalizado: {frame_count} frames analizados")
