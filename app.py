from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io, base64, os, requests

import easyocr
import numpy as np
import re

app = FastAPI()

# Inicializar EasyOCR (Español e Inglés)
reader = easyocr.Reader(['es', 'en'])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "service": "plate-recognition"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs/detect/train/weights/best.pt")

model = YOLO(MODEL_PATH)

# Elementos requeridos (ahora solo la placa)
REQUIRED_ITEMS = ["placa"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img_pil)
    draw = ImageDraw.Draw(img_pil)

    # Fuente para texto
    try:
        font = ImageFont.truetype("arial.ttf", 25)
    except:
        font = ImageFont.load_default()

    results = model.predict(img_pil, conf=0.25)
    boxes = results[0].boxes

    detected = []
    extracted_data = {}

    # --- DIBUJAR PLACA DETECTADA Y EXTRAER TEXTO ---
    for box in boxes:
        cls_id = int(box.cls)
        label = model.names[cls_id]
        
        # Solo procesar si es la clase 'placa' (ajustar si el nombre en el modelo es diferente)
        if label.lower() in ["placa", "license-plate", "license_plate"]:
            detected.append("placa")
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Recortar área de la placa para OCR
            # Añadir un pequeño margen para mejorar el OCR
            margin = 10
            crop = img_np[max(0, y1-margin):min(img_np.shape[0], y2+margin), 
                          max(0, x1-margin):min(img_np.shape[1], x2+margin)]
            
            ocr_result = reader.readtext(crop, detail=0)
            plate_text = "".join(ocr_result).replace(" ", "").upper()
            
            extracted_data["placa"] = plate_text

            # Dibujar cuadro verde y el texto reconocido
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=5)
            draw.text((x1, y1 - 30), f"PLACA: {plate_text}", fill="lime", font=font)

    # Convertir a base64 para previsualización
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    if "placa" in extracted_data:
        message = f"🟢 Placa detectada: {extracted_data['placa']}"
    else:
        message = "🔴 No se detectó ninguna placa"

    return JSONResponse({
        "detected_items": detected,
        "is_detected": "placa" in detected,
        "plate_text": extracted_data.get("placa", ""),
        "message": message,
        "image_base64": img_base64
    })

# --- LÓGICA DE PLACAS OPTIMIZADA ---
def analizar_placa_logic(img_pil):
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
            if label in ["placa", "license-plate", "license_plate", "0"]:
                margin = 5
                crop = img_np[max(0, y1-margin):min(img_np.shape[0], y2+margin), 
                              max(0, x1-margin):min(img_np.shape[1], x2+margin)]
                
                ocr_res = reader.readtext(crop, detail=0)
                
                # Filtrar para quedarse solo con lo que parezca una placa
                # Buscamos la cadena más larga que cumpla con patrones de placas (letras y números)
                plate_candidates = [re.sub(r'[^A-Z0-9]', '', word.upper()) for word in ocr_res]
                
                # Patrones comunes en Colombia: AAA123, AAA12A, AA1234
                # Si algún candidato coincide con un patrón fuerte, lo priorizamos
                regex_patterns = [
                    r'^[A-Z]{3}[0-9]{3}$', # Auto
                    r'^[A-Z]{3}[0-9]{2}[A-Z]$', # Moto
                    r'^[A-Z]{2}[0-9]{4}$' # Público antiguo
                ]
                
                best_match = ""
                for candidate in plate_candidates:
                    if len(candidate) < 5: continue # Muy corto para ser placa
                    
                    # Si coincide con un patrón exacto, lo tomamos de una vez
                    if any(re.match(pattern, candidate) for pattern in regex_patterns):
                        best_match = candidate
                        break
                    
                    # Si no hay patrón exacto, tomamos el más largo que tenga letras y números
                    if len(candidate) > len(best_match):
                        best_match = candidate

                recognized = best_match
                
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

@app.post("/verificar-placa")
async def verificar_placa(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return analizar_placa_logic(img)
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
            return {"error": f"No se pudo descargar la imagen (Status {response.status_code})"}

        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        return analizar_placa_logic(img)
    except Exception as e:
        return {"error": str(e)}
# --- Endpoint rápido para video frame ---
@app.post("/video-frame")
async def video_frame(file: UploadFile = File(...)):
    # Leer bytes → imagen
    img_bytes = await file.read()

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return {
            "error": "Frame inválido",
            "detected": [],
            "missing": REQUIRED_ITEMS,
            "is_complete": False,
            "boxes": []
        }

    # Correr predicción
    results = model.predict(img, imgsz=640, conf=0.25, verbose=False)

    detections = []
    missing = REQUIRED_ITEMS.copy()

    r = results[0]

    # Si YOLO no detecta nada
    if r.boxes is None or len(r.boxes) == 0:
        return {
            "detected": [],
            "missing": REQUIRED_ITEMS,
            "is_complete": False,
            "boxes": []
        }

    # Procesar detecciones
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])

        label = model.names[cls_id]

        detections.append({
            "class": label,
            "confidence": round(conf, 2),
            "bbox": [x1, y1, x2, y2]
        })

        # Quitar de la lista de faltantes
        if label in missing:
            missing.remove(label)

    return {
        "detected": list({d["class"] for d in detections}),  # sin duplicados
        "missing": missing,
        "is_complete": len(missing) == 0,
        "boxes": detections
    }

