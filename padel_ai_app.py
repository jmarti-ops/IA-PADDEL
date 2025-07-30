# padel_ai_app.py
import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
from PIL import Image
from math import degrees, atan2
from ultralytics import YOLO  # Requiere: pip install ultralytics
import os
from fpdf import FPDF
import pandas as pd

st.set_page_config(page_title="Análisis IA Padel", layout="wide")
st.title("🏓 Análisis Inteligente de Pádel con IA")

st.markdown("Sube un video o imagen de un jugador y obtén análisis técnico, postural y de detección de pala/pelota.")

upload_option = st.radio("Selecciona el tipo de archivo", ["Imagen", "Video"], horizontal=True)
uploaded_file = st.file_uploader("Sube una imagen o video", type=["jpg", "png", "mp4"])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

detector_yolo = YOLO("yolov8n.pt")

def calcular_angulo(a, b, c):
    ang = degrees(atan2(c[1]-b[1], c[0]-b[0]) - atan2(a[1]-b[1], a[0]-b[0]))
    return abs(ang + 360) if ang < 0 else ang

def analizar_pose(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)
    kpis = {}
    coords = []
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark
        get_coord = lambda i: (landmarks[i].x, landmarks[i].y)
        hombro = get_coord(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        codo = get_coord(mp_pose.PoseLandmark.LEFT_ELBOW.value)
        muneca = get_coord(mp_pose.PoseLandmark.LEFT_WRIST.value)
        cadera = get_coord(mp_pose.PoseLandmark.LEFT_HIP.value)
        ang_brazo = calcular_angulo(hombro, codo, muneca)
        ang_cadera = calcular_angulo(hombro, cadera, (cadera[0], cadera[1]-0.1))
        kpis = {
            "Ángulo de Brazo (izq)": round(ang_brazo, 1),
            "Ángulo Torso": round(ang_cadera, 1)
        }
        coords.append((int(cadera[0]*frame.shape[1]), int(cadera[1]*frame.shape[0])))
    return frame, result, kpis, coords

def detectar_objetos(frame):
    results = detector_yolo.predict(source=frame, conf=0.3, verbose=False)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = r.names[cls_id].lower()
            if label not in ["person", "sports ball", "racket"]:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {conf:.1f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return frame

def generar_pdf(kpi_resumen, imagen_path=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, "Informe de KPIs - Análisis de Pádel", ln=True, align="C")
    pdf.ln(10)
    for clave, valores in kpi_resumen.items():
        promedio = sum(valores)/len(valores)
        pdf.cell(200, 10, f"{clave}: Promedio = {promedio:.2f}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Resumen:")
    resumen_texto = "Este informe contiene un análisis técnico del gesto deportivo durante un punto de pádel."
    if "Ángulo de Brazo (izq)" in kpi_resumen:
        ang_brazo_avg = sum(kpi_resumen["Ángulo de Brazo (izq)"])/len(kpi_resumen["Ángulo de Brazo (izq)"])
        if ang_brazo_avg > 150:
            resumen_texto += " El ángulo de brazo elevado sugiere una posible extensión excesiva en el golpe."
        elif ang_brazo_avg < 70:
            resumen_texto += " El ángulo de brazo reducido podría limitar la amplitud del swing."
    if "Ángulo Torso" in kpi_resumen:
        ang_torso_avg = sum(kpi_resumen["Ángulo Torso"])/len(kpi_resumen["Ángulo Torso"])
        if ang_torso_avg > 50:
            resumen_texto += " El ángulo de torso alto puede indicar inclinación significativa, revisa el equilibrio."
    resumen_texto += " Valores fuera de rangos esperados podrían indicar necesidad de corrección técnica o física."
    pdf.multi_cell(0, 10, resumen_texto)

    if imagen_path and os.path.exists(imagen_path):
        pdf.image(imagen_path, x=10, y=None, w=180)

    pdf_output = os.path.join(tempfile.gettempdir(), "kpis_padel.pdf")
    pdf.output(pdf_output)
    return pdf_output

if uploaded_file:
    if upload_option == "Imagen":
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        frame = detectar_objetos(frame)
        frame, result, kpis, coords = analizar_pose(frame)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Análisis Visual", use_column_width=True)
        if result.pose_landmarks:
            st.success("✔ Postura detectada correctamente")
            st.write("### KPIs Posturales")
            st.json(kpis)
        else:
            st.warning("⚠️ No se detectó postura. Asegúrate de que el jugador esté claramente visible.")

    elif upload_option == "Video":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        frame_count = 0
        resumen_kpis = []
        heatmap_coords = []
        output_path = os.path.join(tempfile.gettempdir(), "video_salida.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        miniatura_path = os.path.join(tempfile.gettempdir(), "miniatura.jpg")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > 100:
                break
            frame = detectar_objetos(frame)
            frame, result, kpis, coords = analizar_pose(frame)
            if frame_count == 30:
                cv2.imwrite(miniatura_path, frame)
            if coords:
                heatmap_coords.append(coords[0])
            out.write(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)
            if kpis:
                resumen_kpis.append(kpis)
            frame_count += 1

        cap.release()
        out.release()
        st.success(f"🎥 Procesamiento finalizado: {frame_count} frames analizados")

        with open(output_path, "rb") as f:
            st.download_button("⬇️ Descargar video procesado", f, file_name="analisis_padel.avi")

        if resumen_kpis:
            df_kpis = {k: [x[k] for x in resumen_kpis] for k in resumen_kpis[0]}
            for kpi, valores in df_kpis.items():
                st.line_chart(valores, height=150, use_container_width=True)

            pdf_path = generar_pdf(df_kpis, miniatura_path)
            with open(pdf_path, "rb") as fpdf:
                st.download_button("📄 Descargar informe en PDF", fpdf, file_name="informe_kpis.pdf")

            csv_path = os.path.join(tempfile.gettempdir(), "kpis_padel.csv")
            pd.DataFrame(df_kpis).to_csv(csv_path, index=False)
            with open(csv_path, "rb") as fcsv:
                st.download_button("📊 Exportar KPIs a CSV", fcsv, file_name="kpis_padel.csv")

        if heatmap_coords:
            import matplotlib.pyplot as plt
            from scipy.ndimage import gaussian_filter
            heat = np.zeros((height, width))
            for x, y in heatmap_coords:
                if 0 <= x < width and 0 <= y < height:
                    heat[y, x] += 1
            heat = gaussian_filter(heat, sigma=15)
            fig, ax = plt.subplots()
            ax.imshow(heat, cmap="hot", interpolation="nearest")
            ax.set_title("Mapa de calor de posición del jugador")
            st.pyplot(fig)
