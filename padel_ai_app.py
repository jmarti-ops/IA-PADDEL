# padel_ai_app.py
# Este script requiere streamlit y no se puede ejecutar en entornos sin este módulo (como este sandbox)
# Ejecuta esto en local o en Replit/Streamlit Cloud

try:
    import streamlit as st
    import cv2
    import tempfile
    import numpy as np
    import os
    import mediapipe as mp
    from ultralytics import YOLO
    from math import degrees, atan2
    from fpdf import FPDF
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    import urllib.request
    from inference_sdk import InferenceHTTPClient

    st.set_page_config(page_title="Análisis Pádel IA", layout="wide")
    st.title("🏓 Análisis Inteligente de Pádel con IA")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    mp_drawing = mp.solutions.drawing_utils

    upload = st.file_uploader("Sube una imagen o video .mp4", type=["jpg", "png", "mp4"])

    modelo_path = "models/yolov8n.pt"
    modelo_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(modelo_path):
        try:
            st.info("📥 Descargando modelo YOLOv8n...")
            urllib.request.urlretrieve(modelo_url, modelo_path)
            st.success("✅ Modelo YOLO descargado")
        except Exception as e:
            st.error("❌ Error al descargar modelo: " + str(e))

    model = None
    try:
        if os.path.exists(modelo_path):
            model = YOLO(modelo_path)
    except Exception as e:
        st.warning("❌ Error al cargar YOLO local: " + str(e))

    roboflow_client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="YOUR_API_KEY"
    )

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
            l = result.pose_landmarks.landmark
            coord = lambda i: (l[i].x, l[i].y)
            hombro, codo, muneca = coord(11), coord(13), coord(15)
            cadera = coord(23)
            ang_brazo = calcular_angulo(hombro, codo, muneca)
            ang_torso = calcular_angulo(hombro, cadera, (cadera[0], cadera[1]-0.1))
            kpis = {"Ángulo de Brazo (izq)": round(ang_brazo, 1), "Ángulo Torso": round(ang_torso, 1)}
            coords.append((int(cadera[0]*frame.shape[1]), int(cadera[1]*frame.shape[0])))
        return frame, kpis, coords

    def analizar_yolo(frame):
        try:
            if model:
                results = model.predict(frame, imgsz=640, conf=0.3, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            else:
                cv2.imwrite("temp.jpg", frame)
                result = roboflow_client.run_workflow(
                    workspace_name="aixpointpro",
                    workflow_id="custom-workflow",
                    images={"image": "temp.jpg"},
                    use_cache=True
                )
                for pred in result.get("predictions", []):
                    x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']//2), int(pred['height']//2)
                    cv2.rectangle(frame, (x-w, y-h), (x+w, y+h), (255,0,0), 2)
        except Exception as e:
            st.warning(f"⚠️ Error usando YOLO/Roboflow: {e}")
        return frame

    def generar_pdf(kpis):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, "Informe KPIs Pádel IA", ln=True, align="C")
        pdf.ln(10)
        for k, v in kpis.items():
            pdf.cell(200, 10, f"{k}: {v:.2f}", ln=True)
        path = tempfile.mktemp(suffix=".pdf")
        pdf.output(path)
        return path

    if upload:
        if upload.name.endswith(".mp4"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(upload.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            coords, kpi_series = [], []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = analizar_yolo(frame)
                frame, kpis, pts = analizar_pose(frame)
                coords.extend(pts)
                if kpis:
                    kpi_series.append(kpis)
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            cap.release()
            if coords:
                heat = np.zeros(frame.shape[:2])
                for x, y in coords:
                    heat[y, x] += 1
                heat = gaussian_filter(heat, sigma=15)
                fig, ax = plt.subplots()
                ax.imshow(heat, cmap="hot")
                ax.set_title("Mapa de calor")
                st.pyplot(fig)
            if kpi_series:
                df = pd.DataFrame(kpi_series)
                st.line_chart(df)
                pdf = generar_pdf(df.mean())
                with open(pdf, "rb") as f:
                    st.download_button("📄 Descargar Informe PDF", f, file_name="kpis_padel.pdf")
        else:
            img = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR)
            img = analizar_yolo(img)
            frame, kpis, coords = analizar_pose(img)
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Resultado", use_column_width=True)
            st.write(kpis)

except ModuleNotFoundError:
    print("❌ Este script requiere 'streamlit'. Ejecútalo en local o en un entorno como Replit o Streamlit Cloud.")
