from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io, os, requests

app = FastAPI()

# =====================================================
# CORS (restringir en producción)
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import easyocr
import numpy as np

# Inicializar EasyOCR
reader = easyocr.Reader(['es', 'en'])

# =====================================================
# CARGA DE MODELO
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# Si el usuario va a subir su propio modelo, nos aseguramos de que cargue el disponible o de aviso
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ Advertencia: No se encontró 'best.pt'. Se requiere un modelo entrenado para placas.")
    # Usar el modelo base por defecto si no existe el entrenado (opcional)
    model = YOLO("yolo11s.pt")
else:
    model = YOLO(MODEL_PATH)

print("✅ Modelo configurado para placas")

# =====================================================
# FUNCIÓN PRINCIPAL DE PLACAS
# =====================================================
def analizar_placa(img_pil):
    img_np = np.array(img_pil)
    results = model.predict(img_pil, imgsz=640, conf=0.25)

    detections = []
    plate_text = ""

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = model.names.get(cls_id, str(cls_id)).lower()
            
            # Solo si la clase es placa
            if label in ["placa", "license-plate", "license_plate", "0"]: # '0' suele ser el index en muchos datasets de placas
                # OCR del recorte
                margin = 5
                crop = img_np[max(0, y1-margin):min(img_np.shape[0], y2+margin), 
                              max(0, x1-margin):min(img_np.shape[1], x2+margin)]
                
                ocr_res = reader.readtext(crop, detail=0)
                recognized = "".join(ocr_res).replace(" ", "").upper()
                
                if recognized:
                    plate_text = recognized

                detections.append({
                    "class": "placa",
                    "confidence": round(conf, 3),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "text": recognized
                })

    return {
        "is_detected": len(detections) > 0,
        "plate_text": plate_text,
        "detections": detections,
        "message": f"Placa detectada: {plate_text}" if plate_text else "No se detectó el texto de la placa"
    }


# =====================================================
# ENDPOINTS ADAPTADOS
# =====================================================
@app.post("/verificar-placa")
async def verificar_placa(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return analizar_placa(img)
    except Exception as e:
        return {"error": str(e)}

@app.post("/verificar-placa-url")
async def verificar_placa_url(payload: dict):
    try:
        image_url = payload.get("image_url")
        if not image_url:
            return {"error": "Falta image_url"}

        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            return {"error": "No se pudo descargar la imagen"}

        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        return analizar_placa(img)
    except Exception as e:
        return {"error": str(e)}
