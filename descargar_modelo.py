# descargar_modelo.py
# Script para descargar yolov8n.pt en carpeta models/

import os
import urllib.request

MODELO_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
MODELO_PATH = "models/yolov8n.pt"

os.makedirs("models", exist_ok=True)

if not os.path.exists(MODELO_PATH):
    print("Descargando yolov8n.pt...")
    urllib.request.urlretrieve(MODELO_URL, MODELO_PATH)
    print("✅ Modelo descargado en:", MODELO_PATH)
else:
    print("✔ Modelo ya existente en:", MODELO_PATH)
