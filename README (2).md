# Análisis de Pádel con IA 🏓

Esta app permite analizar posturas, golpes y zonas de juego a partir de imágenes o vídeos usando **YOLOv8** y **MediaPipe Pose**.

## Cómo usar

1. Clona este repo:
```bash
git clone https://github.com/tu_usuario/tu_repo.git
cd tu_repo
```

2. Instala dependencias:
```bash
pip install -r requirements.txt
```

3. Descarga el modelo YOLO:
```bash
python descargar_modelo.py
```

4. Lanza la app:
```bash
streamlit run padel_ai_app.py
```

## Requisitos

- Python 3.9–3.11
- `yolov8n.pt` se descarga automáticamente al ejecutar `descargar_modelo.py`
