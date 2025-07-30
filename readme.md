# 📊 Análisis Inteligente de Pádel con IA

Esta aplicación Streamlit permite subir videos o imágenes de partidos de pádel y realizar:

- Detección de jugadores, pelota y raqueta (YOLOv8)
- Análisis postural con MediaPipe (pose)
- Cálculo de KPIs biomecánicos (ej. ángulo de brazo, torso)
- Exportación de video con anotaciones
- Generación de informes PDF + CSV + mapa de calor

---

## 🚀 Instrucciones para desplegar

### 1. Requisitos del sistema

Instala los siguientes paquetes (idealmente en un entorno virtual):

```bash
pip install -r requirements.txt
```

> Usa `opencv-python-headless` en servidores (Streamlit Cloud) y `opencv-python` si corres local.

---

### 2. Ejecutar localmente

```bash
streamlit run padel_ai_app.py
```

---

### 3. Subir a Streamlit Cloud

1. Crea un repositorio en GitHub con estos archivos:

   - `padel_ai_app.py`
   - `requirements.txt`
   - `.gitignore`
   - `README.md`

2. Ve a streamlit.io/cloud

3. Conecta tu repo y lanza la app.

---

## 🧪 Ejemplo de uso

- Sube un video `.mp4` de un partido.
- Verás el video procesado, KPIs, mapa de calor y botones para descargar el informe PDF o CSV.

---

## ⚠️ Notas

- `ultralytics` (YOLOv8) depende de `torch`, por eso se requiere especificar versiones en `requirements.txt`
- Si tienes errores en Cloud, verifica los logs en "Manage app"

---

## 📁 Estructura del repo

```
padel-ai-analyzer/
├── padel_ai_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ✨ Futuras mejoras

- Segmentación de tipos de golpe
- Clasificación de errores técnicos comunes
- Comparación de KPIs entre sesiones

---

Desarrollado con ❤️ por `jmarti-ops`

